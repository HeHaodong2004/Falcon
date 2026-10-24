"""Bounded fitting then independent Layer 2 evaluation jobs, no collection calls."""
import json
from pathlib import Path
import subprocess
import sys
import time
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP


def run_module(module,args,log):
    command=[sys.executable,'-u','-m','eventwm_graph.l2_120k.'+module,*args]
    stream=open(OUT/log,'w',buffering=1)
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
    return process,stream,command


def wait(process,stream,command):
    code=process.wait();stream.close()
    if code:raise RuntimeError('research command failed '+str(code)+': '+repr(command))


def main():
    while not (OUT/'train_prepare.json').exists():time.sleep(10)
    train=json.loads((OUT/'train_prepare.json').read_text());dev=json.loads((OUT/'dev_prepare.json').read_text())
    if train['failures'] or dev['failures']:raise ValueError('unresolved preprocessing failures')
    m=json.loads((SNAP/'manifest.json').read_text())
    if train['roots']!=m['splits']['train']['roots'] or dev['roots']!=m['splits']['dev']['roots']:raise ValueError('incomplete frozen data coverage')
    sealed=dict(created=time.time(),manifest_sha256=sha(SNAP/'manifest.json'),
        artifacts={p:sha(OUT/p) for p in ('train.npy','dev.npy','train_prepare.json','dev_prepare.json','population_prior.json','motion.json')},
        training_roots=train['roots'],trajectory_supervised_roots=train['supervised'],
        dev_roots=dev['roots'],dev_trajectory_supervised_roots=dev['supervised'])
    (OUT/'fit_input_manifest.json').write_text(json.dumps(sealed,indent=2))
    print(json.dumps(dict(stage='full_training_started',inputs=sealed)),flush=True)
    wait(*run_module('learning',['--epochs','50'],'training.log'))
    print(json.dumps(dict(stage='full_training_completed',time=time.time())),flush=True)
    # Independent one-thread workloads; no more than two heavy research jobs.
    evaluation=run_module('evaluate',[],'evaluation.log')
    wait(*run_module('systems',['benchmark'],'systems.log'))
    wait(*run_module('systems',['persistence'],'persistence.log'))
    wait(*run_module('numerics',[],'numerics.log'))
    wait(*run_module('physical',[],'physical.log'))
    wait(*evaluation)
    (OUT/'experiments_completed.json').write_text(json.dumps(dict(time=time.time(),
        selected_checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),fit_inputs=sealed),indent=2))
    print(json.dumps(dict(stage='all_experiments_completed',time=time.time())),flush=True)


if __name__=='__main__':main()
