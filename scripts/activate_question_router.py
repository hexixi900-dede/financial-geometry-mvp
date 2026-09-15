"""Freeze reviewed sources and start one explicitly scoped, resumable queue."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--cohort',choices=['pilot','full','all'],default='pilot')
a=p.parse_args();root=a.root.resolve();release=Path(__file__).resolve().parents[1]
protocol=json.loads((root/'input_protocol.json').read_text())
assert protocol['method_version']=='question_router_v4' and protocol['post_geometry_raw_fallback'] is False
status_path=root/'queue_status.json'
if status_path.exists():
    previous=json.loads(status_path.read_text())
    if previous.get('status')!='complete':raise SystemExit('A queue already exists; inspect it instead of duplicating it')
    if a.cohort!='full' or not (root/'status.txt').read_text().startswith('pilot_complete'):
        raise SystemExit('Only the completed pilot may be advanced to full')
    (root/'pilot_queue_status.json').write_text(status_path.read_text())
manifest_path=root/'release_manifest.json'
if manifest_path.exists():
    manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('Frozen source modified: '+name)
else:
    if any(f.stat().st_size for c in ('pilot','full') for f in (root/c).glob('outcomes.jsonl')):
        raise SystemExit('Refusing to freeze after inference has started')
    files=sorted(release.glob('src/*.py'))+sorted(release.glob('tests/test_*.py'))+sorted(release.glob('scripts/*.py'))
    inputs=[root/'input_protocol.json']+[root/c/name for c in ('full','pilot') for name in ('geometry_inputs.jsonl','gold.jsonl','baseline.jsonl')]
    manifest=dict(method_version='question_router_v4',release=str(release),prepared_at=time.time(),
        source_sha256={str(f.relative_to(release)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        input_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs})
    for directory in (root,release):(directory/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'status.txt').write_text('prepared; frozen '+a.cohort+' queued\n')
command=['python3','-u',str(release/'src/queue_after_experiment.py'),'--release',str(release),'--root',str(root),
         '--entrypoint','run_question_router.py','--cohort',a.cohort,
         '--python','/data/liu_jun/finmme_reproduction/.venv_phase2a/bin/python']
with (root/'queue.log').open('a') as log:
    process=subprocess.Popen(command,cwd=release,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'queue.pid').write_text(str(process.pid)+'\n')
(root.parent/'next_run.json').write_text(json.dumps(dict(root=str(root),release=str(release),method_version='question_router_v4',cohort=a.cohort),indent=2)+'\n')
print(json.dumps(dict(queue_pid=process.pid,cohort=a.cohort,release=str(release),root=str(root))))
