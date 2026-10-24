"""Offline-only readers. Privileged truth never appears in online model methods."""
import argparse,gzip,json,pathlib,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .model import GraphBeliefWorldModel
from .state import normalized

def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def rows(snapshot,split):
    snapshot=pathlib.Path(snapshot);manifest=json.loads((snapshot/'manifest.json').read_text())
    return [(r,snapshot/'roots'/r['root_id']) for r in manifest['rows'] if r['split']==split]

def root_data(folder,include_future_observations=False):
    l=read(folder/'legal_root.json.gz');q=read(folder/'queries.json.gz');o=read(folder/'outcomes.json.gz')
    with np.load(folder/'privileged_future.npz') as f:times=f['timestamps'].copy();positions=f['positions'].copy()
    result={'legal':l,'queries':q,'outcomes':o,'times':times,'truth':positions[..., [0,2]]}
    if include_future_observations:result['nodes']=read(folder/'future_observations.json.gz')
    return result

def population_prior(snapshot):
    counts=np.full(17,.001)
    for _,p in rows(snapshot,'train'):
        with np.load(p/'privileged_future.npz') as f:n=f['positions'].shape[1]
        if n>=len(counts):raise ValueError('population support needs extension')
        counts[n]+=1
    return normalized(counts)

def build(snapshot,output,split,population):
    result=[];start=time.perf_counter();wm=GraphBeliefWorldModel(kind='memory_cv',max_hypotheses=4,population_prior=population)
    for ri,(row,folder) in enumerate(rows(snapshot,split)):
        d=root_data(folder);l=d['legal'];b=wm.initialize(l['history'],l['partial_graph']);h=max(b.hypotheses,key=lambda h:h.weight)
        rel=d['times']-b.timestamp;truth=d['truth'];n=h.n
        mapping=np.full(n,-1,int);current=np.isclose(h.last_seen,b.timestamp)
        visible_indices=np.flatnonzero(current)
        if len(visible_indices) and truth.shape[1]:
            distance=np.linalg.norm(h.mean[visible_indices,None,:2]-truth[None,0],axis=-1);ii,jj=linear_sum_assignment(distance)
            for i,j in zip(ii,jj):
                if distance[i,j]<1e-5:mapping[visible_indices[i]]=j
        targets=np.zeros((n,4,2),np.float32);valid=np.zeros((n,4),bool);horizons=np.array([.5,1.,2.,3.])
        for hi,t in enumerate(horizons):
            if t>rel[-1]+1e-8:continue
            idx=int(np.searchsorted(rel,t));idx=min(idx,len(rel)-1)
            if idx and rel[idx]>t:
                a=(t-rel[idx-1])/(rel[idx]-rel[idx-1]);target=(1-a)*truth[idx-1]+a*truth[idx]
            else:target=truth[idx]
            for i,j in enumerate(mapping):
                if j>=0:targets[i,hi]=target[j]-h.mean[i,:2];valid[i,hi]=True
        result.append({'root_id':row['root_id'],'scene_id':row['scene_id'],'mean':h.mean.astype(np.float32),
                       'covariance':h.covariance.astype(np.float32),'age':(b.timestamp-h.last_seen).astype(np.float32),'graph':b.graph.local_features(h.mean[:,:2]),
                       'targets':targets,'valid':valid,'horizons':horizons.astype(np.float32),'population':truth.shape[1],
                       'unresolved_mass':b.unresolved_mass,'graph_cells':len(b.graph.materialize())})
        if (ri+1)%250==0:print(split,ri+1,'elapsed',round(time.perf_counter()-start),flush=True)
    # Trusted local artifacts only; object archive is intentionally offline-only.
    np.save(output,np.array(result,dtype=object),allow_pickle=True)
    print('complete',split,len(result),'supervised roots',sum(r['valid'].any() for r in result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('output');p.add_argument('--split',choices=['train','dev','holdout'],required=True);p.add_argument('--population');a=p.parse_args()
    if a.population:population=np.asarray(json.loads(pathlib.Path(a.population).read_text()))
    else:
        population=population_prior(a.snapshot);pathlib.Path(a.output+'.population.json').write_text(json.dumps(population.tolist()))
    build(a.snapshot,a.output,a.split,population)
