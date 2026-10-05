#!/usr/bin/env python3
"""Two fresh equivalent native root/branch runs, differing only in retention."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from collector.core import read,write,build_labels,load_wm_input,digest
from collector.storage import compact_workers


def run(output):
    if output.exists(): raise ValueError('Storage smoke requires a new output directory')
    output.mkdir(parents=True)
    records={}; results={}; inputs={}; sizes={}
    for profile in ('validated','production'):
        path=output/profile
        for d in ('roots','legal_observations','queries','branch_truth','labels','audits','diagnostics'):
            (path/d).mkdir(parents=True)
        shutil.copy2(REPO/'datasets/raw/structured_human_castle_v1/roots/replay.json',path/'roots/replay.json')
        for worker in ('root','b023'):
            cmd=[sys.executable,str(REPO/'scripts/eventwm_collect.py'),'worker','--output',str(path),
                 '--worker-id',worker,'--storage-profile',profile]
            if worker!='root': cmd+=['--query-id','q_6d84ea7d1706']
            with (path/'audits'/f'{worker}.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,cwd=REPO)
        b=read(path/'audits/workers/b023/branch.json'); records[profile]=b
        (path/'branch_truth/branches.jsonl').write_text(json.dumps(b,allow_nan=False)+'\n')
        labs=[build_labels(b)]; write(path/'labels/labels.json',labs); results[profile]=labs
        wm=load_wm_input(path,b['query_id']); wm['history_images']='legal_observations/history_images.npz'; inputs[profile]=wm
        if profile=='production': compact_workers(path)
        else:
            with zipfile.ZipFile(path/'legal_observations/branch_images.zip','w') as z:
                z.write(path/'audits/workers/b023/branch_images.npz','b023.npz')
        sizes[profile]=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    a,b=records['validated'],records['production']
    checks=dict(queries_identical=read(output/'validated/queries/queries.json')==read(output/'production/queries/queries.json'),
        entire_branch_identical=a==b,physical_truth_identical=a['truth']==b['truth'],
        labels_identical=results['validated']==results['production'],structured_wm_input_identical=inputs['validated']==inputs['production'],
        root_fingerprint_identical=a['root_fingerprint']==b['root_fingerprint'],
        no_production_branch_images=not list((output/'production').rglob('branch_images*')),
        no_production_worker_raw_copies=not list((output/'production').glob('audits/workers/*/branch.json')),
        materially_smaller=sizes['production']<sizes['validated']*.5)
    with np.load(output/'validated/legal_observations/history_images.npz') as x,np.load(output/'production/legal_observations/history_images.npz') as y:
        checks['root_images_identical']=all(np.array_equal(x[k],y[k]) for k in x.files)
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,bytes=sizes,
                reduction_fraction=1-sizes['production']/sizes['validated'],
                scope='One complete native root prefix and one full populated-observation branch per profile; not a complete collector dataset')
    write(output/'audit.json',report); print(json.dumps(report,indent=2)); assert all(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.absolute())
