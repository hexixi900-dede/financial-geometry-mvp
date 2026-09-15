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


def evidence(source, measurement, no_raw_fallback=False):
    plan = measurement.get('semantic_plan', {})
    targets = plan.get('targets', [])
    values = measurement.get('geometry_values', [])
    ticks = measurement.get('y_axis_ticks', [])
    readings = [dict(target=t, value=compact_value(v), units='same numeric scale as the supplied y-axis ticks')
                for t, v in zip(targets, values)]
    if no_raw_fallback and not values:
        # A question can fail because one target failed while another target
        # was actually measured. Preserve its target index and provenance;
        # a successful local read does not validate the semantic target.
        audits = measurement.get('pipeline_audit') or measurement.get('target_audit') or []
        for index, (target, audit) in enumerate(zip(targets, audits)):
            value = audit.get('auto_local_value', audit.get('value'))
            if audit.get('status') == 'success' and value is not None:
                readings.append(dict(target_index=index, target=target, value=compact_value(value),
                    units='same numeric scale as the supplied y-axis ticks',
                    provenance='completed local pixel measurement; verify target identity in the original image'))
        if not ticks:
            axes = [(audit.get('axis') or {}) for audit in audits]
            axes.append((measurement.get('bar_audit') or {}).get('axis') or {})
            ticks = next((axis['ticks'] for axis in axes if axis.get('ticks')), [])
    return {
        'measurement_status': measurement.get('status'),
        'failure_reason': measurement.get('status_reason') or measurement.get('planner_error') or measurement.get('error'),
        'semantic_plan': plan,
        'target_execution': [
            {'target_index': i, 'status': p.get('status'), 'reason': p.get('status_reason'),
             'grounding': p.get('grounding'), 'series_resolution': p.get('series_resolution')}
            for i, p in enumerate(measurement.get('pipeline_audit') or measurement.get('target_audit') or [])],
        'measurements': readings,
        'axis_tick_texts_and_values': [{'text': t['text'], 'value': t['value']} for t in ticks],
        'operation': plan.get('operation'),
        'python_result': measurement.get('numeric_answer'),
        'calculations': measurement.get('reasoning', {}).get('calculations', []),
        'unit_source': 'Use the visible chart title and axes, supplied answer unit and question; never interpret a currency code prefix as a scale suffix.',
    }


def build_prompt(source, measurement, allow_revision=False, no_raw_fallback=False):
    kind = source['question_type']
    answer_rule = {
        'single_choice': 'This is SINGLE CHOICE. Return answer as exactly one provided letter, e.g. "A". Do not return an array.',
        'multiple_choice': 'This is MULTIPLE CHOICE. Evaluate EVERY option and return all correct letters as a JSON array, e.g. ["A","C"]. Do not stop at one answer.',
        'numerical': 'This is a NUMERICAL question. Return answer as a JSON number in the requested units, without a sentence or option letters.',
    }[kind]
    example = {'single_choice':'A','multiple_choice':['A','C'],'numerical':1.23}[kind]
    response_schema = json.dumps(dict(status='answered' if no_raw_fallback else 'answered|insufficient_evidence', answer=example, evidence_used='brief measurement and unit explanation'))
    feedback = '''
When an overlay is supplied, check the SOLID measured marks against the original chart:
do they identify the requested series and dates/categories, and cover ALL required observations?
An orange proposed box alone is not a measurement. Inspect axis units and the arithmetic order.
For an increase use NEW minus OLD; growth_rate takes OLD then NEW. A monthly yearly mean
requires all months, not one point or the two endpoints. For multiple choice, cover every option.
If the locations and computation support the answer, answer directly; uncertainty from normal
pixel approximation alone is not a reason to discard a usable measurement.
'''
    feedback += ('Clearly distinguish chart-based estimates from values actually measured by the program.\n'
                 if no_raw_fallback else 'Never invent missing values.\n')
    if allow_revision:
        feedback += ('\nIf correcting the plan or location would help, you may return status="revise_plan" with reason '
                     'and a COMPLETE corrected plan. Otherwise answer directly. The program can measure once more.\n'
                     if no_raw_fallback else '\nIf the plan or location is wrong, missing, or incomplete, return status="revise_plan" with reason\n'
                     'and a COMPLETE corrected plan instead of giving up immediately. The program will measure once more.\n')
        feedback += '''The plan has chart_context, chart_type (bar|line|other), y_axis_count, has_direct_value_labels,
operation (direct|difference|growth_rate|ratio|sum|mean|min|max|evidence|unsupported), targets,
calculations and reason. Each target has x_label (actual date/category, never the word "date"),
series (actual series name), position (label|first|last), measurement (value|height|sequence),
and optionally region=[left,top,right,bottom] in 0..1000 relative to the original chart image.
Use an OCR-matchable date without a region when possible. If OCR or series selection failed,
provide a small box around the actual point or a box enclosing the intended bar segment.
For a dated title marking the last observation, use that actual date and position=last.
For floating/waterfall bars, measure height between both ends, not the absolute top coordinate.
For a whole stacked column include all its pieces; for a named segment enclose that segment.
For interval trends use sequence and a box spanning the requested interval, operation=evidence.
For monthly/quarterly averages explicitly list every requested observation with real dates.
direct has one target; difference/ratio/growth_rate have two in their required order.
calculations may be [] or named nested expressions, e.g.
[{"name":"increase","expression":{"op":"difference","args":[{"target":1},{"target":0}]}}].
References use zero-based target indices; no guessed numbers or executable code.
When requesting a correction, return only {"status":"revise_plan","reason":"what to fix","plan":{...complete plan...}}.
Do not copy internal route_geometry/route_reason fields. Do not supply an answer alongside a revision.
'''
    else:
        feedback += ('\nThis is the final pass. No further revision is available. Return your best answer with status=answered, '
                     'using the original chart, question, options and any valid measured evidence.\n'
                     if no_raw_fallback else '\nThis is the final pass: answer from the available evidence or return insufficient_evidence. No further revision is available.\n')
    if measurement.get('status') not in {None, 'success'}:
        if no_raw_fallback:
            feedback += ('\nThe program did not complete a successful measurement for the whole question. '
                         'Any completed local readings are explicitly listed; failed or missing targets have no measured value. '
                         'Check the original image to decide which partial evidence applies. '
                         + ('Request one complete corrected plan if useful; otherwise answer directly from the original chart and any valid evidence.\n'
                            if allow_revision else 'You must still answer from the original chart and any valid evidence. Label any visual estimate as such in evidence_used.\n'))
        else:
            feedback += ('\nThe program has not completed a successful measurement for this question. '
                         'Partial locations may help diagnose the failure, but do not supply a supported final answer. '
                         + ('If you can identify a recoverable target or plan error, request the complete corrected plan.\n'
                            if allow_revision else 'Return insufficient_evidence.\n'))
    if no_raw_fallback:
        policy = '''Answer this financial chart question by combining the original image, question, options and geometry evidence.
Geometry is numerical assistance: the original chart determines the requested target, semantic meaning and units.
Check whether each measured location and calculation matches the question. Use valid measurements; do not copy
values for a wrong target or operation. A failed measurement is not a measured zero and does not prevent answering.
For missing or mismatched evidence, use the original chart to infer your best answer. Clearly identify visual estimates
in evidence_used; never claim they were measured by Geometry. Do not invent a measurement or Python result.
When a plan correction is available and useful, you may request it once. Otherwise give your best answer directly.
In the final pass, always return status=answered even if measurement remains incomplete. The image, question and
options may resolve an answer without a complete numerical measurement. Do not abstain or reuse a cached Raw answer.
'''
    else:
        policy = '''Answer this financial chart question using the supplied geometry measurements.
The numbers below were measured from chart pixels and axis ticks. They are estimates, not ground truth.
First check that the measured locations and calculation match what the question actually asks.
Use matching measurements as numerical evidence, together with the original chart's semantics,
units and options. Do not copy a measurement or Python result that refers to the wrong target
or operation. Request a plan correction when available; otherwise return insufficient_evidence.
Do not replace missing or mismatched measurements with freshly guessed numerical values.
If the available evidence cannot support an answer, request a correction when available and useful; otherwise return status=insufficient_evidence.
'''
    caption = '' if no_raw_fallback else '\nCaption: ' + source.get('caption', '')
    answer_format = ('For an answer, return exactly one JSON object using this format (the answer below is only an example): '
                     if no_raw_fallback else 'Return exactly one JSON object using this format (the answer below is only an example): ')
    return policy + '''
Python has performed the listed arithmetic when a result is present. operation=evidence supplies multiple measurements or sequences for your comparison. Interpret the chart's unit/title and the answer options semantically.
For direct/difference, the result is in axis tick units. For growth_rate, the result is percentage change, already multiplied by 100.
Do not multiply units twice. MYR is a currency code, not M (million). An axis labelled in billions with ticks 60,70,80 has a reading near 70 in billions.
''' + answer_format + response_schema + '''.
For sequences, use their complete pixel samples and x-axis anchors to evaluate intervals/trends; inspect coverage before making claims about the entire interval. Pixel samples are not uniformly spaced observations in time.

''' + feedback + '\nQuestion type: ' + kind + '\n' + answer_rule + caption + '\nRequested answer unit: ' + str(source.get('unit', '')) + '\nQuestion: ' + source['question'] + '\nOptions:\n' + source.get('options', '') + '\nGeometry evidence:\n' + json.dumps(evidence(source, measurement, no_raw_fallback=no_raw_fallback), ensure_ascii=False)


def parse_reply(raw, source):
    try:
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.I)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        if isinstance(obj, list):
            kind = source['question_type']
            if kind not in {'single_choice', 'multiple_choice'}:
                raise ValueError('answer_array_requires_choice_question')
            if kind == 'single_choice' and len(obj) != 1:
                raise ValueError('expected_one_choice')
            obj = {'status': 'answered', 'answer': obj[0] if kind == 'single_choice' else obj,
                   'output_format': 'bare_choice_array'}
        if obj.get('status') == 'revise_plan':
            if not isinstance(obj.get('plan'), dict): raise ValueError('revision_missing_plan')
            return None, 'revision_requested', obj
        if obj.get('status') != 'answered': return None, 'insufficient_evidence', obj
        ans = obj['answer']
        if source['question_type'] in {'single_choice', 'multiple_choice'}:
            allowed = re.findall(r'^\s*([A-Z])\s*[:.)-]', source.get('options',''), re.M)
            if source['question_type'] == 'multiple_choice':
                # Preserve the model's explicit choices despite harmless output
                # formatting; never infer letters from explanatory sentences.
                if isinstance(ans, str):
                    text = ans.strip().upper()
                    if not re.fullmatch(r'[A-Z](?:[\s,;]*[A-Z])*', text):
                        raise ValueError('expected_choice_array_or_letters')
                    ans = re.findall(r'[A-Z]', text)
                if not isinstance(ans, list) or not ans: raise ValueError('expected_choice_array')
                letters = [str(v).strip().upper() for v in ans]
                if any(v not in allowed for v in letters) or len(set(letters)) != len(letters):
                    raise ValueError('invalid_choices')
                ans = ''.join(sorted(letters))
            else:
                if isinstance(ans, list):
                    if len(ans) != 1: raise ValueError('expected_one_choice')
                    ans = ans[0]
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


def official_correct(question_type, prediction, reference, tolerance):
    """Released FinMME scoring: exact choice sets or absolute numeric tolerance.

    Keep the official token/number normalization, including NaN tolerance being
    incorrect, so enhanced predictions use the same rule as official_code_correct.
    """
    text = '' if prediction is None else str(prediction)
    match = re.search(r'(?i)Answer\s*:\s*([^\s\n]+)', text)
    text = match.group(1) if match else text
    for token in ('**', ':', '$\\boxed{', '}$', '\\$', '$', '{', '\\boxed'):
        text = text.replace(token, '')
    reference = '' if reference is None else str(reference)
    if question_type == 'numerical':
        try:
            number = lambda value: float(''.join(c for c in value if c.isdigit() or c in '.-'))
            return abs(number(text) - number(reference)) <= float(tolerance)
        except (ValueError, TypeError, OverflowError):
            return False
    return {c for c in text.upper() if c in 'ABCDEFGHIJKLMN'} == {
        c for c in reference.upper() if c in 'ABCDEFGHIJKLMN'}


def accuracy_summary(records):
    total = len(records)
    raw_correct = sum(r['raw_correct'] for r in records)
    selective_correct = sum(r['selective_correct'] for r in records)
    return dict(questions=total, raw_correct=raw_correct, selective_correct=selective_correct,
                raw_accuracy=raw_correct / total if total else None,
                selective_accuracy=selective_correct / total if total else None,
                delta_pp=100 * (selective_correct - raw_correct) / total if total else None,
                answered=sum(r['route'] == 'geometry_enhanced' for r in records),
                rescued=sum(not r['raw_correct'] and r['selective_correct'] for r in records),
                harmed=sum(r['raw_correct'] and not r['selective_correct'] for r in records))


def evaluate(args):
    no_raw_fallback = bool(getattr(args, 'no_raw_fallback', False))
    inputs = rows(args.inputs)
    gold = {str(r['sample_id']): r for r in rows(args.gold)}
    baseline = {str(r['sample_id']): r for r in rows(args.baseline)}
    replies = {str(r['sample_id']): r for r in rows(args.replies)}
    expected = {str(r['sample_id']) for r in rows(args.prompts)}
    candidates = {str(r['sample_id']): r for r in inputs}
    all_raw = rows(args.raw_metrics)
    raw_by_id = {str(r['sample_id']): r for r in all_raw}
    if not all_raw or len(raw_by_id) != len(all_raw):
        raise ValueError('Full Raw metrics must contain unique, nonempty sample IDs')
    if len(candidates) != len(inputs) or not set(candidates) <= set(raw_by_id):
        raise ValueError('Candidate IDs must be unique and present in the full Raw population')
    if not expected <= set(candidates):
        raise ValueError('Evidence prompts contain IDs outside this candidate cohort')
    if any(int(bool(raw_by_id[sid]['official_code_correct'])) != baseline[sid]['raw_vlm_correct']
           for sid in candidates):
        raise ValueError('Full Raw cache does not match the frozen cohort baseline; refusing a mixed-baseline score')

    # Named metrics already contain real predictions in the existing benchmark.
    # A separate prediction log is optional; score-only caches cannot fabricate it.
    raw_path = getattr(args, 'raw_predictions', None)
    prediction_rows = rows(raw_path) if raw_path is not None else all_raw
    raw_predictions = {str(r['sample_id']): r for r in prediction_rows}
    if raw_path is not None:
        if len(raw_predictions) != len(prediction_rows) or set(raw_predictions) != set(raw_by_id):
            raise ValueError('Raw predictions must cover exactly the full Raw metric population')
        for sid, prediction_row in raw_predictions.items():
            metric = raw_by_id[sid]
            if 'prediction' in metric and prediction_row.get('prediction') != metric['prediction']:
                raise ValueError(f'Raw prediction/metric mismatch for sample {sid}')
            if 'reference' in prediction_row and ('tolerance' in prediction_row or
                    'tolerance' in (prediction_row.get('metadata') or {}) or
                    metric['question_type'] != 'numerical'):
                tolerance = prediction_row.get('tolerance', (prediction_row.get('metadata') or {}).get('tolerance'))
                actual = official_correct(metric['question_type'], prediction_row.get('prediction'),
                                          prediction_row['reference'], tolerance)
                if actual != bool(metric['official_code_correct']):
                    raise ValueError(f'Raw official score mismatch for sample {sid}')
    predictions_available = all('prediction' in raw_predictions.get(sid, {}) for sid in raw_by_id)

    measurement_path = getattr(args, 'measurements', None)
    if measurement_path is None:
        adjacent = args.replies.with_name('measurements.jsonl')
        measurement_path = adjacent if adjacent.exists() else None
    measurements = {str(r['sample_id']): r for r in rows(measurement_path)} if measurement_path else {}
    if no_raw_fallback:
        if measurement_path is None:
            raise ValueError('--no-raw-fallback requires measurements with explicit Geometry entry decisions')
        invalid_entry = [sid for sid, m in measurements.items() if sid in candidates
                         and type(m.get('geometry_entered')) is not bool]
        if invalid_entry:
            raise ValueError('No-Raw-fallback measurements require boolean geometry_entered: '
                             + ', '.join(sorted(invalid_entry)[:10]))
    unmeasured = set(candidates) - set(measurements) if measurement_path else set()
    # A successful measurement that has not yet been written as a prompt/reply
    # is still pending, as is a candidate whose planner/measurement has not run.
    waiting_evidence = {sid for sid, m in measurements.items()
                        if sid in candidates and (
                            m.get('geometry_entered') is True if no_raw_fallback else
                            (m.get('status') == 'success' and
                             m.get('semantic_plan', {}).get('has_direct_value_labels') is False))}
    pending_ids = unmeasured | ((expected | waiting_evidence) - set(replies))
    evaluation_status = ('partial' if pending_ids else
                         'complete' if measurement_path else 'completion_unverified')

    full_records = []
    for metric in all_raw:
        sid = str(metric['sample_id'])
        reply = replies.get(sid, {})
        measurement = measurements.get(sid, {})
        success = (sid in candidates and sid in expected and
                   reply.get('status') == 'success' and reply.get('prediction') is not None)
        geometry_entered = bool(sid in candidates and
                                measurement.get('geometry_entered', sid in expected))
        must_use_geometry = no_raw_fallback and geometry_entered
        raw_correct = int(bool(metric['official_code_correct']))
        enhanced_correct = 0
        if success:
            answer = gold[sid]
            enhanced_correct = int(official_correct(metric['question_type'], reply['prediction'],
                                                    answer['reference'], answer.get('tolerance')))
        status = ('outside_candidate' if sid not in candidates else
                  'pending' if sid in pending_ids else reply.get('status', 'raw_route'))
        route = ('geometry_enhanced' if success else
                 'geometry_pending' if must_use_geometry and sid in pending_ids else
                 'geometry_failed' if must_use_geometry else 'raw')
        raw_prediction_available = 'prediction' in raw_predictions.get(sid, {})
        result = dict(sample_id=sid, question_type=metric['question_type'],
                      is_candidate=sid in candidates, geometry_entered=geometry_entered,
                      status=status, route=route, score_provisional=sid in pending_ids,
                      raw_correct=raw_correct, hybrid_correct=enhanced_correct,
                      selective_correct=enhanced_correct if success or must_use_geometry else raw_correct,
                      raw_prediction_available=raw_prediction_available,
                      prediction_available=success or (not must_use_geometry and raw_prediction_available))
        if raw_prediction_available:
            result['raw_prediction'] = raw_predictions[sid]['prediction']
        if success:
            result['prediction'] = reply['prediction']
        elif must_use_geometry:
            result['prediction'] = None
            result['geometry_failure_reason'] = (
                'answer_pending' if sid in pending_ids else
                reply.get('status') or measurement.get('status_reason') or
                measurement.get('status') or 'missing_final_answer')
        elif raw_prediction_available:
            result['prediction'] = result['raw_prediction']
        full_records.append(result)

    records = [r for r in full_records if r['is_candidate']]
    summary = accuracy_summary(records)
    hard = sum(r['hybrid_correct'] for r in records)
    failure_counts = {}
    for record in records:
        if record['route'] == 'geometry_failed':
            reason = record['geometry_failure_reason']
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    summary.update(
        raw_fallback_policy=('disabled_after_geometry_entry' if no_raw_fallback else
                             'cached_raw_on_unusable_evidence'),
        geometry_entered=sum(r['geometry_entered'] for r in records),
        geometry_failed=sum(r['route'] == 'geometry_failed' for r in records),
        geometry_pending=sum(r['route'] == 'geometry_pending' for r in records),
        geometry_failure_counts=failure_counts,
        raw_routed=sum(r['route'] == 'raw' and not r['score_provisional'] for r in records),
        score_provisional=bool(pending_ids),
        pending_score_definition=(
            'Provisional: unprocessed candidates retain Raw; entered Geometry awaiting a final answer count as zero.'
            if no_raw_fallback else
            'Provisional: candidates without a usable final answer retain Raw.'))
    summary.update(evaluation_status=evaluation_status, prompts=len(expected),
                   replies=len(set(replies) & set(candidates)), pending=len(pending_ids),
                   pending_measurements=len(unmeasured) if measurement_path else None,
                   pending_answers=len((expected | waiting_evidence) - set(replies)),
                   measurement_completion_verified=measurement_path is not None,
                   hard_correct=hard, hard_accuracy=hard / len(records) if records else None,
                   score_definition='official_code_correct: exact choice sets; absolute per-question numeric tolerance',
                   accuracy_scale='fraction', overall_definition='all-question micro accuracy, not the mean of three type accuracies',
                   raw_prediction_source=str(raw_path or args.raw_metrics),
                   full_prediction_exported=predictions_available)
    full = accuracy_summary(full_records)
    summary.update(full_questions=full['questions'], full_raw_correct=full['raw_correct'],
                   full_selective_correct=full['selective_correct'], full_raw_accuracy=full['raw_accuracy'],
                   full_selective_accuracy=full['selective_accuracy'], full_delta_pp=full['delta_pp'],
                   full_finmme_population=full['questions'] == 11099)
    for key, population in [('by_question_type', records), ('full_by_question_type', full_records)]:
        summary[key] = {kind: accuracy_summary([r for r in population if r['question_type'] == kind])
                        for kind in sorted({r['question_type'] for r in population})}
    summary['main_table'] = []
    for method, metric_key in [('Raw Qwen2.5-VL', 'raw_accuracy'),
                               ('Geometry-enhanced Qwen2.5-VL', 'selective_accuracy')]:
        row = dict(method=method, Overall=full[metric_key])
        for kind, label in [('single_choice', 'Single'), ('multiple_choice', 'Multi'), ('numerical', 'Calculation')]:
            row[label] = summary['full_by_question_type'].get(kind, {}).get(metric_key)
        summary['main_table'].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name('per_sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    full_path = args.output.with_name('full_predictions.jsonl')
    if predictions_available:
        full_path.write_text(''.join(json.dumps({**r, 'evaluation_status': evaluation_status}) + '\n'
                                    for r in full_records))
    elif full_path.exists():
        full_path.unlink()  # Do not leave an earlier export masquerading as this evaluation.
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--inputs',type=Path,required=True);prep.add_argument('--measurements',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    runner=sub.add_parser('run');runner.add_argument('--inputs',type=Path,required=True);runner.add_argument('--model',type=Path,required=True);runner.add_argument('--output',type=Path,required=True);runner.add_argument('--limit',type=int)
    ev=sub.add_parser('evaluate')
    for flag in ['inputs','gold','baseline','replies','prompts','raw-metrics','output']:ev.add_argument('--'+flag,type=Path,required=True)
    ev.add_argument('--raw-predictions',type=Path,help='Optional full Raw prediction log; otherwise use real prediction fields in --raw-metrics')
    ev.add_argument('--measurements',type=Path,help='Candidate measurement log for end-to-end completion tracking; defaults beside --replies')
    ev.add_argument('--no-raw-fallback', action='store_true', help='Once Geometry is entered, score only its final VLM answer; unusable final answers are wrong rather than replaced by cached Raw')
    args=p.parse_args();{'prepare':prepare,'run':run,'evaluate':evaluate}[args.command](args)

if __name__=='__main__':main()
