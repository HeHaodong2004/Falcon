"""Freeze source, extract frozen Stage01 data, train once, evaluate and time."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'output/stage2_graphdelta_20261024'
PYTHON='/research-user/miniforge3/envs/eventwm-falcon/bin/python'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files={'eventwm_graph/stage2/'+p.name:p.read_bytes() for p in (BASE/'eventwm_graph/stage2').glob('*.py')}
    files['eventwm_graph/partial/map.py']=(BASE/'eventwm_graph/partial/map.py').read_bytes()
    files['eventwm_graph/__init__.py']=b'"""Isolated Stage2 research namespace."""\n'
    files['eventwm_graph/partial/__init__.py']=b'from .map import PartialGraph, UNKNOWN, KNOWN_FREE, KNOWN_OCCUPIED\n'
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();frozen=OUT/'frozen_sources'/identity
    if not frozen.exists():
        for name,data in files.items():
            path=frozen/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);path.chmod(0o444)
        (frozen/'manifest.json').write_text(json.dumps(dict(identity=identity,files=hashes),indent=2)+'\n')
        (frozen/'manifest.json').chmod(0o444)
    (OUT/'source_path.txt').write_text(str(frozen)+'\n')
    protocol=dict(schema='stage2_fixed_supervised_protocol_1',source_identity=identity,epochs=6,seed=2410,width=48,batch_size=8192,
        learning_rate=.001,weight_decay=.0001,gradient_clip=5.,height_huber_weight=.2,
        positive_samples_per_query=64,unknown_samples_per_query=64,sampling='inverse inclusion; equal Root/within-Root Query weight',
        training='all final eligible Roots and all Query branches; no new simulation',
        evaluation='all original2110 development Roots/108 scenes and every Query; scene bootstrap2000 seed2411',
        decision='raw probability argmax; no validation selection or threshold tuning',
        scope='newly observed initially UNKNOWN static C-space cells only; known-map corrections counted as unsupported',
        collection='MANUALLY_STOPPED; no future collection or Stage1 calibration work',stage1_required_as_input=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=dict(os.environ,EVENTWM_STAGE2_BASE=str(BASE),PYTHONPATH=str(frozen),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
    def run(module,args,log):
        cmd=[PYTHON,'-m','eventwm_graph.stage2.'+module]+args
        print(json.dumps(dict(command=cmd,source=str(frozen))),flush=True)
        with (OUT/log).open('a') as stream:subprocess.run(cmd,cwd='/tmp',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not (OUT/'prepared/train/manifest.json').exists():run('data',['--split','train','--workers','20'],'prepare_train.log')
    checkpoint=OUT/'training/final/last.pt'
    if not (OUT/'training/final/complete.json').exists():
        if checkpoint.parent.exists():raise RuntimeError('incomplete fit needs diagnosis; no silent repeated training')
        run('train',['--name','final','--epochs','6','--batch-size','8192','--device','cuda:0'],'train.log')
    if not (OUT/'evaluation/final_dev/metrics.json').exists():run('evaluate',['--checkpoint',str(checkpoint)],'evaluate.log')
    if not (OUT/'benchmarks/final/metrics.json').exists():run('benchmark',['--checkpoint',str(checkpoint)],'benchmark.log')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_identity=identity))+'\n')


if __name__=='__main__':main()
