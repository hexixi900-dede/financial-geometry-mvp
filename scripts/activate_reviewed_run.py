"""Queue the prepared release; replace only an identified, empty GPU waiter."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--release',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--previous-root',type=Path,required=True)
a=p.parse_args();release=a.release.resolve();root=a.root.resolve();old=a.previous_root.resolve()
if (root/'queue_status.json').exists() or (root/'queue.pid').exists():
    raise SystemExit('New queue already exists; inspect instead of creating a duplicate')
protocol=json.loads((root/'input_protocol.json').read_text())
version=protocol['method_version']
assert version=='visual_feedback_v4' and protocol['post_geometry_raw_fallback'] is False
old_queue=json.loads((old/'queue_status.json').read_text());old_release=Path(old_queue['release'])
def own_process(pid):
    try:parts=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return False
    return str(old).encode() in parts and any(str(old_release).encode() in part for part in parts)
pids=[int(old_queue[k]) for k in ('queue_pid','worker_pid') if old_queue.get(k) and own_process(int(old_queue[k]))]
files=sorted(release.glob('src/*.py'))+sorted(release.glob('tests/test_*.py'))+sorted(release.glob('scripts/*.py'))
manifest=dict(method_version=version,release=str(release),prepared_at=time.time(),
              source_sha256={str(f.relative_to(release)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
for dest in (root,release):(dest/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'status.txt').write_text('prepared; waiting for predecessor\n')
command=['python3','-u',str(release/'src/queue_after_experiment.py'),'--release',str(release),
         '--root',str(root),'--entrypoint','run_visual_feedback.py',
         '--python','/data/liu_jun/finmme_reproduction/.venv_phase2a/bin/python']
for pid in pids:command+=['--after-pid',str(pid)]
with (root/'queue.log').open('a') as log:
    proc=subprocess.Popen(command,cwd=release,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'queue.pid').write_text(str(proc.pid)+'\n')
def no_inference():
    names=('plans','measurements_initial','feedback_initial','revision_plans','measurements_revision','feedback_final','replies','measurements')
    return not any(path.stat().st_size for c in ('pilot','full') for name in names
                   if (path:=old/c/(name+'.jsonl')).exists())
replaced=False;paused=[]
try:
    if pids and no_inference() and (old/'status.txt').read_text().startswith('waiting_for_gpu'):
        for pid in pids:
            if own_process(pid):
                try:os.kill(pid,signal.SIGSTOP);paused.append(pid)
                except ProcessLookupError:pass
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
        gpu_pids={int(s.strip()) for s in gpu.stdout.splitlines() if s.strip().isdigit()}
        if (gpu.returncode==0 and no_inference() and
            (old/'status.txt').read_text().startswith('waiting_for_gpu') and not gpu_pids.intersection(paused)):
            for pid in paused:
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            replaced=True
finally:
    for pid in paused:
        try:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
if replaced:
    old_queue.update(status='superseded_before_inference',successor=str(root),updated_at=time.time())
    (old/'queue_status.json').write_text(json.dumps(old_queue,indent=2)+'\n')
    (old/'status.txt').write_text(f'superseded_before_inference; {version} queued\n')
(root.parent/'next_run.json').write_text(json.dumps(dict(root=str(root),release=str(release),method_version=version,predecessor=str(old)),indent=2)+'\n')
(root/'handoff.json').write_text(json.dumps(dict(predecessor=str(old),predecessor_pids=pids,replaced_empty_waiter=replaced,queue_pid=proc.pid),indent=2)+'\n')
print(json.dumps(dict(queue_pid=proc.pid,replaced_empty_waiter=replaced,root=str(root))))
