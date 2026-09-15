"""Prepare a fresh run using the same VLM-visible information as the frozen Raw run."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write_rows(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))


def matched_input(source, raw):
    for key in ('sample_id', 'question', 'question_type', 'options'):
        if str(source.get(key, '')) != str(raw.get(key, '')):
            raise ValueError(f'{source["sample_id"]}: Raw {key} mismatch')
    image = Path(raw['image_path'])
    if not image.is_file():
        raise FileNotFoundError(image)
    result = dict(source)
    # The existing geometry JPEG is an uncropped thumbnail of this PNG. Keep its
    # calibrated pixel coordinates; normalized VLM regions map to either size.
    result.update(caption='', vlm_image_path=str(image),
                  vlm_image_sha256=raw['image_sha256'], input_protocol='raw_direct_matched_v1',
                  unit=(raw.get('metadata', {}).get('unit') or '')
                  if source['question_type']=='numerical' else '')
    return result


def prepare(source_root, output_root, raw_path):
    if source_root.resolve()==output_root.resolve():
        raise ValueError('The active experiment must not be changed')
    raw_rows=rows(raw_path)
    raw={str(r['sample_id']):r for r in raw_rows}
    if len(raw)!=len(raw_rows):raise ValueError('Duplicate Raw sample IDs')
    settings={(r['inference_parameters']['min_pixels'],r['inference_parameters']['max_pixels']) for r in raw_rows}
    if len(settings)!=1:raise ValueError('Raw image settings are not uniform')
    min_pixels,max_pixels=settings.pop()
    prepared={}
    for cohort in ('pilot','full'):
        prepared[cohort]=[matched_input(r,raw[str(r['sample_id'])])
                          for r in rows(source_root/cohort/'geometry_inputs.jsonl')]
    output_root.mkdir(parents=True,exist_ok=False)
    for cohort,records in prepared.items():
        out=output_root/cohort;out.mkdir()
        for name in ('geometry_inputs','planner_inputs'):write_rows(out/(name+'.jsonl'),records)
        for name in ('gold.jsonl','baseline.jsonl'):
            shutil.copy2(source_root/cohort/name,out/name)
        summary=dict(questions=len(records),question_types=dict(Counter(r['question_type'] for r in records)),
                     input_protocol='raw_direct_matched_v1',inference_results_copied=False)
        (out/'prepare_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copy2(source_root/'full/chart_geometry_inputs.jsonl',output_root/'full/chart_geometry_inputs.jsonl')
    for name in ('full_question_audit.csv','full_chart_audit.csv'):
        if (source_root/'full'/name).exists():shutil.copy2(source_root/'full'/name,output_root/'full'/name)
    protocol=dict(name='raw_direct_matched_v1',raw_predictions=str(raw_path.resolve()),
                  raw_predictions_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  full_questions=len(raw_rows),candidate_questions=len(prepared['full']),
                  min_pixels=min_pixels,max_pixels=max_pixels,extra_caption=False,
                  requested_unit='Raw metadata.unit for numerical questions',
                  vlm_image='Exact per-question PNG used by Raw',
                  geometry_image='Existing uncropped JPEG thumbnail; normalized regions preserve spatial alignment',
                  source_run=str(source_root.resolve()),reuses_old_inference=False)
    (output_root/'input_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output_root/'status.txt').write_text('prepared; waiting for predecessor experiment to finish\n')
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--raw-predictions',required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(prepare(args.source_root,args.output_root,args.raw_predictions),indent=2))
