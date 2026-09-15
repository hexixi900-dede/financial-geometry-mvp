"""Wait without using the GPU, then start a prepared experiment automatically."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def process_identity(pid):
    try:
        # starttime distinguishes the original worker from a subsequently reused PID.
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        return stat[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--after-pid',type=int,action='append',default=[])
    p.add_argument('--entrypoint',default='run_unified_resident.py',choices=['run_unified_resident.py','run_visual_feedback.py','run_question_router.py'])
    p.add_argument('--cohort',choices=['pilot','full','all'],default='pilot')
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--python',required=True)
    args=p.parse_args()
    root=args.root.resolve();release=args.release.resolve()
    with (root/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        predecessors={str(pid):process_identity(pid) for pid in args.after_pid}
        queue=dict(queue_pid=os.getpid(),predecessors=predecessors,release=str(release),root=str(root),entrypoint=args.entrypoint,
                   status='waiting_for_predecessor',created_at=time.time())
        def save(status):
            queue.update(status=status,updated_at=time.time())
            (root/'queue_status.json').write_text(json.dumps(queue,indent=2)+'\n')
        save('waiting_for_predecessor')
        while any(identity is not None and process_identity(int(pid))==identity
                  for pid,identity in predecessors.items()):
            time.sleep(2)
        save('starting_successor')
        protocol=json.loads((root/'input_protocol.json').read_text())
        pointer=dict(root=str(root),release=str(release),method_version=protocol['method_version'])
        temp=root.parent/'current_run.json.tmp';temp.write_text(json.dumps(pointer,indent=2)+'\n')
        temp.replace(root.parent/'current_run.json')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        command=[args.python,'-u',str(release/'src'/args.entrypoint),'--root',str(root)]
        if args.entrypoint=='run_question_router.py':command+=['--cohort',args.cohort]
        with (root/'run.log').open('a') as log:
            while True:
                child=subprocess.Popen(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT)
                queue['worker_pid']=child.pid;save('running_successor')
                code=child.wait()
                queue['last_exit_code']=code
                if code!=75:break
                save('waiting_for_gpu');time.sleep(5)
        save('complete' if code==0 else 'failed')
        if code:
            (root/'status.txt').write_text(f'failed (exit {code}); see run.log\n')
        raise SystemExit(code)


if __name__=='__main__':main()
