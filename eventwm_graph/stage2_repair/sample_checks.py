"""Check recovered quantized sample keys themselves and inverse inclusion."""
from collections import defaultdict
import hashlib
import numpy as np
from .diagnose import OUT,PREP,HF,read,save
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def main():
    meta=read(OUT/'trace.json')['examples'];wanted={(e['root_id'],e['start_node'],e['end_node']):i for i,e in enumerate(meta,1)};result=[]
    m=read(PREP/'pilot_fit_mixed_ray/manifest.json')
    for shard in m['shards']:
        chosen=[e for e in shard['segments']if(e['root_id'],e['start_node'],e['end_node'])in wanted]
        if not chosen:continue
        with np.load(shard['path'],allow_pickle=False)as z:x=z['x'];y=z['y'];w=z['weight']
        for e in chosen:
            i=wanted[(e['root_id'],e['start_node'],e['end_node'])]
            with np.load(OUT/'examples'/('%02d.npz'%i),allow_pickle=False)as z:keys=z['keys'];gt=z['y'];path=z['path'];res=z['geometry'][0]
            seed=int(hashlib.sha256(('segment-2424:'+e['root_id']+str(e['start_node'])+':'+str(e['end_node'])).encode()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed);ids=[];expected_weights=[]
            for stratum in [np.flatnonzero(gt>0),np.flatnonzero(gt==0)]:
                n=min(48,len(stratum))
                if n:ids.extend(rng.choice(stratum,n,replace=False));expected_weights.extend([len(stratum)/(n*len(keys)*e['segments_in_root'])]*n)
            sl=slice(e['start'],e['end']);centers=x[sl,36:38].astype(np.float64)*10+path[0,[1,3]];recovered=np.floor(centers/res).astype(np.int32)
            np.testing.assert_array_equal(recovered,keys[ids]);np.testing.assert_array_equal(y[sl],gt[ids]);np.testing.assert_allclose(w[sl],expected_weights,atol=1e-9,rtol=1e-6)
            result.append(dict(example=i,samples=len(ids),exact_recovered_keys=True,exact_labels=True,weight_sum=float(w[sl].sum()),expected_weight_sum=1/e['segments_in_root'],max_coordinate_error_m=float(np.max(abs(centers-(keys[ids]+.5)*res)))))
    # Byte-identical point features across the real small set: count collisions,
    # including their class pairs. These are effective-input collisions only.
    xx=[];yy=[]
    for i in range(1,len(meta)+1):
        with np.load(OUT/'examples'/('%02d_features.npz'%i),allow_pickle=False)as z:xx.append(z['x']);yy.append(z['y'])
    x=np.concatenate(xx);y=np.concatenate(yy);raw=np.ascontiguousarray(x).view(np.dtype((np.void,x.shape[1]*4))).ravel();unique,inverse,counts=np.unique(raw,return_inverse=True,return_counts=True)
    mask=np.zeros((len(unique),3),bool);mask[inverse,y]=True;conflicts=mask.sum(1)>1
    save(OUT/'sample_checks.json',dict(examples=result,feature_rows=len(y),unique_effective_inputs=len(unique),conflicting_effective_inputs=int(conflicts.sum()),rows_in_conflicting_groups=int(counts[conflicts].sum()),scope='Exact F32 deployed 88-feature collisions on selected training windows only; not a global task ceiling.'))
    print('sample identities',len(result),'conflicting effective inputs',int(conflicts.sum()),flush=True)


if __name__=='__main__':main()
