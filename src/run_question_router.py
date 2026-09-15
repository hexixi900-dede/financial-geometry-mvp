"""Question decision -> target plan -> modular measurements -> evidence answer.

Only a negative question decision selects the frozen Raw answer. Capability and
measurement failures never change the decision. Every stage is resumable.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

import vlm_need_router as router
import framework_plan as planner
from run_unified_resident import Runtime, append, invoke, load_log, merge_progress, write_rows
from vlm_evidence_answerer import rows, parse_reply

METHOD_VERSION = 'question_router_v4'
JOURNALS = ('routers', 'plans', 'measurements_initial', 'feedback_initial',
            'revision_plans', 'measurements_revision', 'feedback_final',
            'prompts', 'replies', 'measurements', 'outcomes')


def audit(source):
    return dict(sample_id=str(source['sample_id']), chart_id=source['chart_id'],
                method_version=METHOD_VERSION, input_protocol=source.get('input_protocol'),
                vlm_image_sha256=source.get('vlm_image_sha256'))


def route_question(runtime, args, source):
    msg = router.messages(source, args.min_pixels, args.max_pixels)
    attempts = []
    for attempt in range(2):
        raw, error = invoke(runtime, msg, 384)
        decision = None
        if not error:
            try:
                decision = router.normalize_router(router.extract_json(raw))
            except (ValueError, TypeError, AttributeError, RecursionError) as exc:
                error = str(exc)
        attempts.append(dict(raw_text=raw, error=error, messages=msg))
        if decision is not None:
            return {**audit(source), **decision, 'status': 'success', 'attempts': attempts}
        # Correct only the output format; the source image and question stay the same.
        msg = router.messages(source, args.min_pixels, args.max_pixels) + [
            {'role': 'assistant', 'content': raw},
            {'role': 'user', 'content': 'The decision could not be parsed: ' + str(error) +
             '. ' + router.FORMAT_INSTRUCTION}]
    return {**audit(source), 'needs_geometry': None, 'status': 'router_error', 'attempts': attempts,
            'reason': 'No valid question decision; counted as an error, never routed to Raw.'}


def plan_targets(runtime, args, source):
    msg = planner.messages(source, args.min_pixels, args.max_pixels)
    raw, error = invoke(runtime, msg, 3072)
    plan = {}
    if not error:
        try:
            plan = planner.normalize_plan(planner.extract_json(raw), source['question'])
        except (ValueError, TypeError, AttributeError, RecursionError) as exc:
            error = str(exc)
    return {**audit(source), 'status': 'success' if plan else 'plan_parse_failed',
            'plan': plan, 'plan_valid': bool(plan), 'error': error, 'raw_text': raw,
            'messages': msg, 'planner_version': METHOD_VERSION}


def failed_measurement(source, record, reason):
    return {**audit(source), 'status': 'failed', 'capability_status': 'not_evaluated',
            'measurement_status': 'none', 'status_reason': reason,
            'semantic_plan': record.get('plan') or {}, 'geometry_values': [],
            'numeric_answer': None, 'reasoning': {}, 'pipeline_audit': [],
            'planner_error': record.get('error'), 'planner_raw_text': record.get('raw_text', '')}


def measure_cpu(args):
    """CPU child: one OCR reader per batch, sharing chart preparation within it."""
    from geometry_toolbox import measure_plan
    plans = load_log(args.plans)
    completed = load_log(args.output)
    reader = None
    cache = {}
    for source in rows(args.inputs):
        sid = str(source['sample_id'])
        if sid in completed:
            continue
        record = plans[sid]
        if record['status'] != 'success':
            result = failed_measurement(source, record, 'semantic_plan_missing_or_invalid')
        else:
            if reader is None:
                import easyocr
                import torch
                torch.set_num_threads(4)
                reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=args.ocr_models,
                                        download_enabled=False, verbose=False)
            try:
                result = measure_plan(reader, source, record['plan'], cache=cache)
                result.update(audit(source))
            except (ValueError, OSError, ArithmeticError) as exc:
                result = failed_measurement(source, record, 'geometry_error: ' + str(exc))
        append(args.output, result)
        print(json.dumps(dict(stage='measurement', sample_id=sid, status=result['status'])), flush=True)


def measure(args, out, sources, plan_name, output_name):
    completed = load_log(out / (output_name + '.jsonl'))
    pending = [s for s in sources if str(s['sample_id']) not in completed]
    if pending:
        inputs = out / (output_name + '_inputs.jsonl')
        write_rows(inputs, pending)
        subprocess.run([args.ocr_python, '-u', str(Path(__file__).resolve()), '--root', str(args.root),
                        '--measure-only', '--inputs', str(inputs), '--plans', str(out / (plan_name + '.jsonl')),
                        '--output', str(out / (output_name + '.jsonl')), '--ocr-models', args.ocr_models],
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
    return load_log(out / (output_name + '.jsonl'))


def review(runtime, args, source, measurement, out, attempt):
    from geometry_visual_feedback import render_measurement_overlay
    from framework_answer import messages, evidence
    overlay = out / 'overlays' / (str(source['sample_id']) + f'_{attempt}.png')
    render_measurement_overlay(source, measurement, overlay)
    msg = messages(source, measurement, str(overlay), args.min_pixels, args.max_pixels,
                   allow_revision=attempt == 0)
    tokens = 3072 if attempt == 0 else 768
    raw, error = invoke(runtime, msg, tokens)
    answer, status, parsed = (None, 'inference_error', {}) if error else parse_reply(raw, source)
    if attempt and status == 'revision_requested':
        answer, status = None, 'revision_limit_reached'
    prompt = '\n'.join(part['text'] for m in msg if isinstance(m.get('content'), list)
                       for part in m['content'] if part.get('type') == 'text')
    return {**audit(source), 'status': status, 'prediction': answer, 'raw_reply': raw,
            'parsed_reply': parsed, 'prompt': prompt, 'messages': msg,
            'evidence': evidence(source, measurement), 'overlay_path': str(overlay), 'attempt': attempt,
            'model': str(args.model), 'generation': dict(do_sample=False, max_new_tokens=tokens,
                min_pixels=args.min_pixels, max_pixels=args.max_pixels), 'error': error}


def run_cohort(args, cohort, runtime):
    from evaluate_question_router import evaluate
    out = args.root / cohort
    sources = rows(out / 'geometry_inputs.jsonl')
    logs = {name: load_log(out / (name + '.jsonl')) for name in JOURNALS}
    pending = [s for s in sources if str(s['sample_id']) not in logs['outcomes']]

    def save(name, record):
        sid = str(record['sample_id'])
        if sid not in logs[name]:
            append(out / (name + '.jsonl'), record)
            logs[name][sid] = record

    def status(text):
        (args.root / 'status.txt').write_text(cohort + ': ' + text + '\n')

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        status('question-level routing')
        entered = []
        for source in batch:
            sid = str(source['sample_id'])
            if sid not in logs['routers']:
                save('routers', route_question(runtime, args, source))
            decision = logs['routers'][sid]
            if decision['needs_geometry'] is True:
                entered.append(source)
            else:
                route = 'raw' if decision['needs_geometry'] is False else 'router_error'
                save('outcomes', {**audit(source), 'route': route, 'status': decision['status'],
                                  'prediction': None, 'needs_geometry': decision['needs_geometry']})
            print(json.dumps(dict(stage='route', sample_id=sid, needs_geometry=decision['needs_geometry'])), flush=True)
        status('target planning')
        for source in entered:
            if str(source['sample_id']) not in logs['plans']:
                save('plans', plan_targets(runtime, args, source))
        status('initial CPU geometry')
        logs['measurements_initial'] = measure(args, out, entered, 'plans', 'measurements_initial')
        status('review image and measured evidence')
        revisions = []
        for source in entered:
            sid = str(source['sample_id'])
            if sid not in logs['feedback_initial']:
                save('feedback_initial', review(runtime, args, source, logs['measurements_initial'][sid], out, 0))
            first = logs['feedback_initial'][sid]
            if first['status'] == 'revision_requested':
                if sid not in logs['revision_plans']:
                    plan, error = {}, None
                    try:
                        plan = planner.normalize_plan(first['parsed_reply']['plan'], source['question'])
                    except (KeyError, ValueError, TypeError, AttributeError, RecursionError) as exc:
                        error = str(exc)
                    save('revision_plans', {**audit(source), 'plan': plan, 'plan_valid': bool(plan),
                        'status': 'success' if plan else 'plan_parse_failed', 'error': error,
                        'raw_text': first['raw_reply'], 'revision_source': 'visual_feedback'})
                revisions.append(source)
        if revisions:
            status('corrected CPU geometry')
            logs['measurements_revision'] = measure(args, out, revisions, 'revision_plans', 'measurements_revision')
        status('final evidence answers')
        for source in entered:
            sid = str(source['sample_id'])
            first = logs['feedback_initial'][sid]
            revised = first['status'] == 'revision_requested'
            m = logs['measurements_revision'][sid] if revised else logs['measurements_initial'][sid]
            if revised or first['status'] != 'success':
                if sid not in logs['feedback_final']:
                    save('feedback_final', review(runtime, args, source, m, out, 1))
                reply = logs['feedback_final'][sid]
            else:
                reply = first
            reply = {**reply, 'geometry_entered': True, 'plan_revised': revised}
            save('prompts', {**audit(source), 'prompt': reply['prompt'], 'messages': reply['messages'],
                             'overlay_path': reply['overlay_path']})
            save('replies', reply)
            save('measurements', {**m, 'geometry_entered': True, 'plan_revised': revised,
                                  'feedback_status': reply['status'], 'needs_geometry': True})
            # This is the terminal commit. It never reads the cached Raw prediction.
            save('outcomes', {**audit(source), 'route': 'framework', 'needs_geometry': True,
                               'prediction': reply['prediction'], 'status': reply['status']})
            print(json.dumps(dict(stage='answer', sample_id=sid, status=reply['status'], revised=revised)), flush=True)
        evaluate(out, args.raw_metrics, args.raw_predictions)
    for name in JOURNALS:
        (out / (name + '.jsonl')).touch(exist_ok=True)
    evaluate(out, args.raw_metrics, args.raw_predictions)


def verify_release(args):
    protocol = json.loads((args.root / 'input_protocol.json').read_text())
    if protocol.get('method_version') != METHOD_VERSION or protocol.get('post_geometry_raw_fallback') is not False:
        raise ValueError('Wrong experiment protocol')
    if (args.min_pixels, args.max_pixels) != (protocol['min_pixels'], protocol['max_pixels']):
        raise ValueError('Image resolution must match Raw')
    for field in ('raw_metrics', 'raw_predictions'):
        source = Path(getattr(args, field))
        if str(source) != protocol[field] or hashlib.sha256(source.read_bytes()).hexdigest() != protocol[field + '_sha256']:
            raise ValueError('Frozen Raw comparison changed: ' + field)
    release = Path(__file__).resolve().parents[1]
    manifest = json.loads((args.root / 'release_manifest.json').read_text())
    if manifest['method_version'] != METHOD_VERSION or manifest['release'] != str(release):
        raise ValueError('Wrong frozen release')
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((release / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source modified: ' + name)
    for name, expected in manifest.get('input_sha256', {}).items():
        if hashlib.sha256((args.root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen input modified: ' + name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cohort', choices=['pilot', 'full', 'all'], default='pilot')
    p.add_argument('--model', default='/data/liu_jun/blind_cssa/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--ocr-python', default='/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python')
    p.add_argument('--ocr-models', default='/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models')
    p.add_argument('--raw-metrics', default='/data/liu_jun/finmme_reproduction/outputs/metrics/qwen25vl7b_direct_full_per_sample.jsonl')
    p.add_argument('--raw-predictions', default='/data/liu_jun/finmme_reproduction/outputs/predictions/qwen25vl7b_direct_full.jsonl')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--min-pixels', type=int, default=200704)
    p.add_argument('--max-pixels', type=int, default=802816)
    p.add_argument('--gpu-poll-seconds', type=float, default=10)
    p.add_argument('--measure-only', action='store_true')
    p.add_argument('--inputs', type=Path)
    p.add_argument('--plans', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.measure_only:
        measure_cpu(args)
        return
    if args.batch_size < 1:
        raise ValueError('Invalid batch size')
    verify_release(args)
    with (args.root / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = Runtime(args.model, args.root / 'status.txt', args.min_pixels, args.max_pixels, args.gpu_poll_seconds)
        if args.cohort in {'pilot', 'all'}:
            run_cohort(args, 'pilot', runtime)
            for name in JOURNALS:
                merge_progress(args.root / 'pilot' / (name + '.jsonl'), args.root / 'full' / (name + '.jsonl'))
        if args.cohort in {'full', 'all'}:
            run_cohort(args, 'full', runtime)
        else:
            from evaluate_question_router import evaluate
            evaluate(args.root / 'full', args.raw_metrics, args.raw_predictions)
        (args.root / 'status.txt').write_text('complete\n' if args.cohort != 'pilot' else 'pilot_complete; full not started\n')


if __name__ == '__main__':
    main()
