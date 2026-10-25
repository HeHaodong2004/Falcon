"""PRIVILEGED DIAGNOSTIC ONLY: isolate missing static-support knowledge.

Combines legally measured partial-range visibility with true scene support.
This is an error-source ablation, never a deployable model or acceptance score.
"""
from collections import defaultdict
import time
import numpy as np
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .geometry_reference import asset_path
from .full_evaluate import codes
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


def run():
    tick=time.perf_counter();m=read(OUT/'range_evidence_dev/manifest.json')
    raw,_=source_rows('dev');raw={r['root_id']:r for r in raw}
    assets={}
    for p in (OUT/'geometry/assets').glob('*/identity.json'):
        receipt=read(p);assets[receipt['identity']['navmesh_sha256']]=p.parent/'arrays.npz'
    cache={};scenes=defaultdict(list);pooled=np.zeros((3,3,3),np.int64);gates=np.zeros((3,3,2,2),np.int64);rows=[]
    for row in m['records']:
        rid=row['root_id'];private=member(raw[rid],'privileged_initialization.json.gz')
        navhash=sha(asset_path(private['episode']['info']['graph_navmesh']))
        if navhash not in cache:
            with np.load(assets[navhash])as z:
                cache[navhash]=(z['keys'],z['targets'][:,0,1]-.025)
        supportkeys,supportheights=cache[navhash]
        packed=read(HF/'packed/dev_pilot'/(rid+'.json'));a=load_arrays(packed)
        lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
        with np.load(row['path'])as z:extra=z['extra'];y=z['y']
        with np.load(row['original_feature_path'])as z:x=z['x']
        cms=[]
        for e in row['segments']:
            v=get_segment(a,lookup[e['start_node'],e['end_node']]);p=v[3];sl=slice(e['start'],e['end'])
            keys=np.floor((x[sl,20:22].astype(np.float64)*10+p[0,[1,3]])/a['geometry'][0]).astype(np.int32)
            support=np.unique(codes(supportkeys[np.abs(supportheights-p[0,2])<=.18]))
            query=codes(keys);at=np.searchsorted(support,query);valid=at<len(support)
            gt_support=valid&(support[np.minimum(at,len(support)-1)]==query)if len(support)else np.zeros(len(keys),bool)
            ev=extra[sl];truth=y[sl];mode=np.where(ev[:,0]>0,0,np.where(ev[:,1]>0,1,2))
            np.add.at(gates,(truth.astype(int),mode,gt_support.astype(int),(ev[:,3]>0).astype(int)),1)
            hard=[np.where(ev[:,1]>0,2,np.where(ev[:,0]>0,1,0)),
                  np.where(ev[:,1]>0,2,np.where((ev[:,0]>0)&gt_support,1,0)),
                  np.where(ev[:,1]>0,2,np.where((ev[:,0]>0)&gt_support&(ev[:,6]>0),1,0))]
            cm=np.array([np.bincount(truth.astype(int)*3+h,minlength=9).reshape(3,3)for h in hard])
            # Preserve all out-of-domain positives from the original universe.
            missing=~np.isin(codes(v[4]),query,assume_unique=False)
            for state in v[5][missing]:cm[:,state,0]+=1
            cms.append(cm);pooled+=cm
        mean=np.mean(cms,0);scenes[row['scene_id']].append(mean)
        rows.append(dict(root_id=rid,scene_id=row['scene_id'],windows=len(cms),confusion=mean.tolist()))
    mean=np.mean([np.mean(values,0)for values in scenes.values()],0)
    names=['partial_geometry_no_support_gate','ORACLE_support_with_partial_visibility','ORACLE_support_and_legal_obstacle_clearance']
    result=dict(role='PRIVILEGED ERROR-SOURCE DIAGNOSTIC ONLY; NOT DEPLOYABLE OR ACCEPTANCE',
                hierarchical={n:extended_rates(c)for n,c in zip(names,mean)},
                pooled={n:extended_rates(c)for n,c in zip(names,pooled)},pooled_confusion=pooled.tolist(),
                gates=gates.tolist(),gate_axes=['true UNKNOWN/FREE/BLOCKED','partial-geometry FREE/BLOCKED/NONE','oracle_support_false/true','measured_floor_support_false/true'],
                roots=len(rows),windows=sum(r['windows']for r in rows),records=rows,seconds=time.perf_counter()-tick)
    save(OUT/'support_oracle_ablation.json',result)
    print({n:(v['label_aware_delta_iou'],v['revelation_iou'])for n,v in result['hierarchical'].items()},flush=True)


if __name__=='__main__':run()
