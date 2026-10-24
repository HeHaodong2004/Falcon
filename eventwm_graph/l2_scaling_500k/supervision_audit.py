"""Measure inherited supervision ambiguity/censoring without changing a target."""
import collections
import json
import time
import numpy as np
from .common import OUT,sha,dump

def run():
    start=time.perf_counter()
    while not (OUT/'preparation.json').exists():
        if time.perf_counter()-start>10800:raise RuntimeError('Preparation did not finish')
        time.sleep(1)
    rows=np.load(OUT/'train.npy',allow_pickle=True);preparation=json.loads((OUT/'preparation.json').read_text());groups={}
    for name,part in (('original',rows[:3212]),('added',rows[3212:]),('all',rows)):
        ages=[];duration=[];ambiguous=[];duplicates=[];observations=0;unique_ids=set()
        for row in part:
            with np.load(OUT/'snapshot/roots'/row['root_id']/'privileged_future.npz') as f:truth=f['positions'][0][:,[0,2]]
            root_ambiguous=False;root_duplicate=False;duration.append(float(row['times'][-1]-row['times'][0]))
            for c in row['components']:
                ages.extend(c['age'][c['selected']].tolist());assigned=[]
                for p in c['truth'][0]:
                    distance=np.linalg.norm(truth-p,axis=-1);matching=np.flatnonzero(distance<2e-4)
                    root_ambiguous|=len(matching)>1
                    assigned.append(int(distance.argmin()));observations+=1
                root_duplicate|=len(assigned)!=len(set(assigned))
            if root_ambiguous:ambiguous.append(dict(root_id=row['root_id'],scene_id=row['scene_id']))
            if root_duplicate:duplicates.append(dict(root_id=row['root_id'],scene_id=row['scene_id']))
        groups[name]=dict(supervised_roots=len(part),scenes=len({r['scene_id'] for r in part}),
            selected_age_quantiles=np.quantile(ages,[0,.5,.95,1]).tolist(),
            truth_duration_quantiles=np.quantile(duration,[0,.05,.5,.95,1]).tolist(),
            roots_with_truth_duration_at_least={str(H):sum(d>=H-1e-8 for d in duration) for H in (.5,1.,2.,3.,4.)},
            ambiguous_physical_correspondence_roots=len(ambiguous),ambiguous_scenes=len({r['scene_id'] for r in ambiguous}),
            ambiguity_ledger=ambiguous,duplicate_target_roots=len(duplicates),duplicate_ledger=duplicates,
            selected_component_records=observations)
    assert len(rows)==preparation['supervised_roots']
    dump(OUT/'supervision_audit.json',dict(groups=groups,train_sha256=sha(OUT/'train.npy'),seconds=time.perf_counter()-start,
        semantics='Diagnostic only: all inherited nearest-match targets remain unchanged, including ambiguous correspondences. Component/mode records are not independent roots. Gap weighting changes future target times, not root observations or the input age of selected root-visible people.'))

if __name__=='__main__':run()
