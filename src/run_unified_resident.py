"""One Qwen load for visual planning, CPU measurement and evidence answers, with resume."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from vlm_semantic_planner import messages, normalize_plan, extract_json, gpu_is_idle
from vlm_evidence_answerer import rows, build_prompt, evidence, parse_reply, foreign_gpu_processes


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def load_log(path):
    """Recover only an interrupted final write in our own append-only logs."""
    if not path.exists():return {}
    data=path.read_bytes();offset=0;out={}
    lines=data.splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:r=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1 and not line.endswith(b'\n'):
                path.write_bytes(data[:offset]);break
            raise
        out[str(r['sample_id'])]=r;offset+=len(line)
    else:
        if data and not data.endswith(b'\n'):
            with path.open('ab') as f:f.write(b'\n')
    return out


def merge_progress(source, destination):
    saved=load_log(destination)
    for sid,row in load_log(source).items():
        if sid not in saved:append(destination,row)


def write_rows(path, records):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    temp.replace(path)


class Runtime:
    def __init__(self, model_path, status_path=None, min_pixels=200704, max_pixels=802816, gpu_poll_seconds=5):
        self.model_path=model_path;self.model=None;self.status_path=status_path
        self.min_pixels=min_pixels;self.max_pixels=max_pixels;self.gpu_poll_seconds=gpu_poll_seconds

    def generate(self, msg, max_tokens):
        if self.model is None:
            stage=self.status_path.read_text() if self.status_path and self.status_path.exists() else 'visual planning\n'
            while not gpu_is_idle():
                if self.status_path:self.status_path.write_text('waiting_for_gpu; model not loaded\n')
                print(json.dumps({'time':time.time(),'gpu_idle':False}),flush=True);time.sleep(self.gpu_poll_seconds)
            if self.status_path:self.status_path.write_text('loading_model\n')
            os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            import torch
            from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
            self.processor=AutoProcessor.from_pretrained(self.model_path,local_files_only=True,min_pixels=self.min_pixels,max_pixels=self.max_pixels)
            self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
            print('Qwen loaded once; continuing through pilot and full run.',flush=True)
            if self.status_path:self.status_path.write_text(stage)
        if foreign_gpu_processes():raise SystemExit(75)
        import torch
        from qwen_vl_utils import process_vision_info
        text=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        images,_=process_vision_info(msg)
        encoded=None;generated=None
        try:
            encoded=self.processor(text=[text],images=images,padding=True,return_tensors='pt')
            if encoded['input_ids'].shape[1]+max_tokens>32768:
                raise ValueError('evidence_exceeds_32k_context; no silent truncation')
            encoded=encoded.to('cuda:0')
            with torch.inference_mode():generated=self.model.generate(**encoded,max_new_tokens=max_tokens,do_sample=False)
            return self.processor.batch_decode(generated[:,encoded['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
        finally:
            del encoded,generated


def invoke(runtime, msg, tokens):
    try:return runtime.generate(msg,tokens),None
    except (ValueError,OSError) as exc:return '',str(exc)
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():raise
        import torch
        torch.cuda.empty_cache()
        return '',str(exc)


def run_cohort(args, cohort, runtime):
    out=args.root/cohort
    min_pixels=getattr(args,'min_pixels',200704);max_pixels=getattr(args,'max_pixels',802816)
    source=rows(out/'geometry_inputs.jsonl')
    plans=load_log(out/'plans.jsonl');measured=load_log(out/'measurements.jsonl');replies=load_log(out/'replies.jsonl')
    pending=[s for s in source if str(s['sample_id']) not in measured or
             (measured[str(s['sample_id'])].get('status')=='success' and str(s['sample_id']) not in replies)]
    for start in range(0,len(pending),args.batch_size):
        batch=pending[start:start+args.batch_size]
        (args.root/'status.txt').write_text(f'{cohort}: visual_planning batch {start//args.batch_size+1}; model retained between stages\n')
        for src in batch:
            sid=str(src['sample_id'])
            if sid in plans:continue
            msg=messages(src,min_pixels,max_pixels)
            raw,error=invoke(runtime,msg,1536)
            plan={};status='inference_error' if error else 'success'
            if not error:
                try:plan=normalize_plan(extract_json(raw),src['question'])
                except (ValueError,TypeError,AttributeError,RecursionError) as exc:status='plan_parse_failed';error=str(exc)
            record=dict(sample_id=sid,chart_id=src['chart_id'],plan=plan,status=status,error=error,
                        raw_text=raw,messages=msg,planner_version='visual_regions_matched_input_v1',
                        input_protocol=src.get('input_protocol'),vlm_image_sha256=src.get('vlm_image_sha256'))
            append(out/'plans.jsonl',record);plans[sid]=record
            print(json.dumps({'stage':'plan','cohort':cohort,'sample_id':sid,'status':status}),flush=True)
        (args.root/'status.txt').write_text(f'{cohort}: CPU geometry; model retained for next answers\n')
        if any(str(s['sample_id']) not in measured for s in batch):
            write_rows(out/'batch_inputs.jsonl',batch)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
            subprocess.run([args.ocr_python,'-u','src/phase8_hybrid_qa.py','run-geometry',
                '--inputs',str(out/'batch_inputs.jsonl'),'--plans',str(out/'plans.jsonl'),
                '--cached-charts',str(args.root/'full'/'chart_geometry_inputs.jsonl'),
                '--model-dir',args.ocr_models,'--output',str(out/'measurements.jsonl'),
                '--repair-geometry','--evidence-only','--threads','4'],env=env,check=True)
            measured=load_log(out/'measurements.jsonl')
        prompts=[]
        for src in source:
            m=measured.get(str(src['sample_id']),{})
            if m.get('status')=='success' and m.get('semantic_plan',{}).get('has_direct_value_labels') is False:
                prompts.append({**src,'prompt':build_prompt(src,m),'evidence':evidence(src,m)})
        write_rows(out/'prompts.jsonl',prompts)
        (args.root/'status.txt').write_text(f'{cohort}: evidence answers\n')
        for rec in prompts:
            sid=str(rec['sample_id'])
            if sid in replies:continue
            msg=[{'role':'system','content':'Use the measured evidence to answer. Output JSON only.'},
                 {'role':'user','content':[{'type':'image','image':Path(rec.get('vlm_image_path') or rec['image_path']).resolve().as_uri(),
                  'min_pixels':min_pixels,'max_pixels':max_pixels},{'type':'text','text':rec['prompt']}]}]
            raw,error=invoke(runtime,msg,512)
            answer,status,parsed=(None,'inference_error',{}) if error else parse_reply(raw,rec)
            result=dict(sample_id=sid,chart_id=rec['chart_id'],status=status,prediction=answer,raw_reply=raw,
                        parsed_reply=parsed,prompt=rec['prompt'],messages=msg,evidence=rec['evidence'],
                        model=str(args.model),generation={'do_sample':False,'max_new_tokens':512,
                        'min_pixels':min_pixels,'max_pixels':max_pixels},input_protocol=rec.get('input_protocol'),
                        vlm_image_sha256=rec.get('vlm_image_sha256'),error=error)
            append(out/'replies.jsonl',result);replies[sid]=result
            print(json.dumps({'stage':'answer','cohort':cohort,'sample_id':sid,'status':status}),flush=True)
    for name in ('plans','measurements','prompts','replies'):(out/(name+'.jsonl')).touch(exist_ok=True)
    evaluation=['python3','src/vlm_evidence_answerer.py','evaluate',
        '--inputs',str(out/'geometry_inputs.jsonl'),'--gold',str(out/'gold.jsonl'),'--baseline',str(out/'baseline.jsonl'),
        '--replies',str(out/'replies.jsonl'),'--prompts',str(out/'prompts.jsonl'),
        '--measurements',str(out/'measurements.jsonl'),
        '--raw-metrics',args.raw_metrics,'--output',str(out/'summary.json')]
    if getattr(args,'raw_predictions',None):evaluation.extend(['--raw-predictions',str(args.raw_predictions)])
    subprocess.run(evaluation,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('phase9_evidence/unified'))
    p.add_argument('--model',default='/data/liu_jun/blind_cssa/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--ocr-python',default='/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python')
    p.add_argument('--ocr-models',default='/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models')
    p.add_argument('--raw-metrics',default='/data/liu_jun/finmme_reproduction/outputs/metrics/qwen25vl7b_direct_full_per_sample.jsonl')
    p.add_argument('--raw-predictions',default='/data/liu_jun/finmme_reproduction/outputs/predictions/qwen25vl7b_direct_full.jsonl')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--min-pixels',type=int,default=200704)
    p.add_argument('--max-pixels',type=int,default=802816)
    p.add_argument('--gpu-poll-seconds',type=float,default=5)
    args=p.parse_args()
    protocol_path=args.root/'input_protocol.json'
    if not protocol_path.exists():raise ValueError('Prepare a separate matched-input run before launching this runtime')
    protocol=json.loads(protocol_path.read_text())
    if (args.min_pixels,args.max_pixels)!=(protocol['min_pixels'],protocol['max_pixels']):
        raise ValueError('VLM image limits must match the frozen Raw input protocol')
    if args.gpu_poll_seconds<=0:raise ValueError('gpu-poll-seconds must be positive')
    with (args.root/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=Runtime(args.model,args.root/'status.txt',args.min_pixels,args.max_pixels,args.gpu_poll_seconds)
        run_cohort(args,'pilot',runtime)
        for name in ('plans','measurements','replies'):
            merge_progress(args.root/'pilot'/(name+'.jsonl'),args.root/'full'/(name+'.jsonl'))
        run_cohort(args,'full',runtime)
        (args.root/'status.txt').write_text('complete\n')

if __name__=='__main__':main()
