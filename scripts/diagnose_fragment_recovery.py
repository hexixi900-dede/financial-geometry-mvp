"""Isolate pixel extraction using frozen plans/OCR, without inference or scoring."""
import json
from pathlib import Path
import cv2
from geometry_toolbox import measure_targets
b=Path('/data/liu_jun/financial_geometry_mvp')
old=b/'phase9_evidence/question_router_v3/pilot'
output=b/'phase9_evidence/question_router_v4'
output.mkdir(exist_ok=True)
sources={r['sample_id']:r for r in map(json.loads,(old/'geometry_inputs.jsonl').read_text().splitlines())}
results=[]
for old_m in map(json.loads,(old/'measurements.jsonl').read_text().splitlines()):
    chart=old_m.get('chart_audit') or {}
    if chart.get('status')!='success':continue
    sid=old_m['sample_id']
    image=cv2.imread(sources[sid]['image_path'])
    result=measure_targets(None,image,chart,old_m['semantic_plan'])
    def brief(m):
        return dict(status=m['status'],values=m.get('geometry_values'),targets=[
            dict(label=a['semantic_target']['x_label'],status=a['status'],
                 reason=a.get('status_reason'),geometry=a.get('auto_local_geometry'))
            for a in m.get('target_audit',[])])
    row=dict(sample_id=sid,old=brief(old_m),new=brief(result))
    results.append(row)
    print(json.dumps(dict(sample_id=sid,old_count=old_m.get('measurement_success_count',0),
        new_count=result.get('measurement_success_count',0),old_values=old_m.get('geometry_values'),
        new_values=result.get('geometry_values')),ensure_ascii=False),flush=True)
    if sid=='1491':
        from geometry_visual_feedback import render_measurement_overlay
        render_measurement_overlay(sources[sid],result,output/'diagnostic_1491.png')
(output/'fragment_replay.json').write_text(json.dumps(dict(
    formal_accuracy_result=False,source='Frozen v3 plans/OCR; new pixels only. No Gold read.',
    records=results),ensure_ascii=False,indent=2))
