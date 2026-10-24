"""Reuse an exact completed prefix and prepare a disjoint tail at <=2 jobs.

Only a specifically identified research preprocessing process can be stopped.
The collector is neither imported nor controlled. Partitioning changes compute
scheduling, not roots, splits, features, probabilities, or supervision targets.
"""
import argparse
import json
import os
import pickle
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP


def join(prefix_pid,cut=6000):
    root=Path.cwd().resolve();prefix_path=Path('/proc')/str(prefix_pid)
    def verify_prefix_process():
        argv=(prefix_path/'cmdline').read_bytes().split(b'\0')
        argv=[x.decode() for x in argv if x]
        if argv[-3:]!=['-m','eventwm_graph.l2_120k.data','train'] or (prefix_path/'cwd').resolve()!=root:
            raise ValueError('PID is not the exact original research data-preparation command/worktree')
        return argv
    verified=verify_prefix_process()
    manifest=json.loads((SNAP/'manifest.json').read_text());expected=[r['root_id'] for r in manifest['rows'] if r['split']=='train']
    # Wait until the development process has actually published its complete
    # metadata, so the new tail never creates a third heavy research job.
    while not (OUT/'dev_prepare.json').exists():time.sleep(10)
    devmeta=json.loads((OUT/'dev_prepare.json').read_text())
    if devmeta['failures']:raise ValueError('development preprocessing failures require explicit repair')
    log=open(OUT/'train_tail_prepare.log','w',buffering=1)
    command=['nice','-n','19','ionice','-c','3',sys.executable,'-u','-m','eventwm_graph.l2_120k.data',
             'train','--start-index',str(cut),'--tag','tail']
    tail=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(tail_pid=tail.pid,command=command,expected_tail_roots=len(expected)-cut)),flush=True)
    started=time.time();prefix=None
    while prefix is None:
        if tail.poll() not in (None,0):raise RuntimeError('tail preprocessing failed')
        try:
            candidate=np.load(OUT/'train_partial.npy',allow_pickle=True).tolist()
            if len(candidate)>=cut:
                if len(candidate)!=cut or [r['root_id'] for r in candidate]!=expected[:cut]:
                    raise ValueError('prefix checkpoint root identities differ from exact partition')
                prefix=candidate
        except (FileNotFoundError,EOFError,OSError,pickle.UnpicklingError):pass
        if prefix is None:time.sleep(10)
    np.save(OUT/'train_prefix.npy',np.array(prefix,dtype=object),allow_pickle=True)
    verified=verify_prefix_process()
    # The exact cut is durable in our own independent copy before stopping the
    # original redundant continuation. No collector PID or process group signal.
    os.kill(prefix_pid,signal.SIGTERM)
    print(json.dumps(dict(research_prefix_stopped=prefix_pid,completed_roots=cut,command_verified=verified)),flush=True)
    code=tail.wait();log.close()
    if code:raise RuntimeError('tail preprocessing failed')
    tailrows=np.load(OUT/'train_tail.npy',allow_pickle=True).tolist()
    if [r['root_id'] for r in tailrows]!=expected[cut:]:raise ValueError('tail identities differ')
    combined=prefix+tailrows
    if [r['root_id'] for r in combined]!=expected or len(set(expected))!=len(expected):raise ValueError('root coverage mismatch')
    np.save(OUT/'train_joined.npy',np.array(combined,dtype=object),allow_pickle=True)
    os.replace(OUT/'train_joined.npy',OUT/'train.npy')
    for name in ('train.npy','dev.npy'):os.chmod(OUT/name,0o444)
    report=dict(roots=len(combined),built=len(combined),supervised=sum(bool(r['components']) for r in combined),
        failures=[],partition_cut=cut,partition_method='exact completed prefix6000 + disjoint deterministic tail; <=2 heavy jobs',
        stopped_research_preparation_pid=prefix_pid,verified_command=verified,
        source_manifest_sha256=sha(SNAP/'manifest.json'),train_cache_sha256=sha(OUT/'train.npy'),
        dev_cache_sha256=sha(OUT/'dev.npy'),population_sha256=sha(OUT/'population_prior.json'),
        motion_sha256=sha(OUT/'motion.json'),tail_metadata=json.loads((OUT/'train_tail_prepare.json').read_text()),
        join_workflow_seconds=time.time()-started,
        production_processes_controlled=False,root_order_and_exact_coverage_verified=True)
    (OUT/'train_prepare.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix-pid',type=int,required=True);p.add_argument('--cut',type=int,default=6000);join(**vars(p.parse_args()))
