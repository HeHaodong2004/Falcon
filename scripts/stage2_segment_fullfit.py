"""Await complete causal extraction, audit exact weights, fit once, evaluate."""
from pathlib import Path
import json
import subprocess
import sys
import time
import numpy as np
from eventwm_graph.stage2_segment.common import OUT,BASE,dump,sha,snapshot


def run(module,args,log):
    with (OUT/'logs'/log).open('x') as f:
        subprocess.run([sys.executable,str(BASE/'scripts/stage2_segment_run.py'),module,*args],cwd=BASE,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    path=OUT/'prepared/full_mixed_ray/manifest.json'
    while not path.exists() or not (OUT/'root_seal_audit_corrected/complete.json').exists():time.sleep(10)
    m=json.loads(path.read_text());seal=json.loads((OUT/'root_seal_audit_corrected/complete.json').read_text())
    assert seal['passed'] and m['requested_roots']==105838
    rootweights={};segments=0;rows=0;tick=time.perf_counter()
    for shard in m['shards']:
        assert sha(shard['path'])==shard['sha256']
        with np.load(shard['path'],allow_pickle=False) as a:
            w=a['weight'];y=a['y'];x=a['x'];h=a['height'];assert x.shape==(len(y),104)
            assert np.all(x[:,:16]==0) and np.isfinite(x).all() and np.isfinite(h).all() and np.isfinite(w).all() and np.all(w>=0)
            assert np.all((y==0)|(y==1)|(y==2))
            for ex in shard['segments']:
                weight=float(np.sum(w[ex['start']:ex['end']],dtype=np.float64));rootweights[ex['root_id']]=rootweights.get(ex['root_id'],0)+weight
                segments+=1
            rows+=len(y)
    assert len(rootweights)==m['roots'] and rows==m['sampled_cells']
    maximum=max(abs(v-1.) for v in rootweights.values());assert maximum<2e-6
    dump(OUT/'full_fit_integrity_gate.json',dict(passed=True,roots=len(rootweights),segments=segments,rows=rows,manifest_sha256=sha(path),
        every_shard_hash_verified=True,all_feature_ranges_and_labels_checked=True,max_root_weight_mass_error=maximum,
        exclusions=m['failures'],source=snapshot(),seconds=time.perf_counter()-tick))
    run('eventwm_graph.stage2_segment.train',['--split','full','--strategy','mixed','--features','ray','--name','final'],'train_final.log')
    # Evaluation arguments/source are fixed before launching; no model selection.
    run('eventwm_graph.stage2_segment.evaluate',['--name','final_segment_dev','--checkpoint',str(OUT/'training/final/last.pt'),'--workers','8'],'evaluate_final.log')
    dump(OUT/'FULL_FIT_AND_EVALUATION_COMPLETE.json',dict(checkpoint_sha256=sha(OUT/'training/final/last.pt')))


if __name__=='__main__':main()
