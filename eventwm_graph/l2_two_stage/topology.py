"""Observed-edge diffusion features; no scene ID, mesh, or hidden topology."""
import argparse
import collections
import gzip
import json
import re
import time
from pathlib import Path
import numpy as np
from .common import OUT,dump,sha


def topology_features(graph,points):
    nodes=graph['global_nodes'];ids=sorted(nodes);n=len(ids);shape=points.shape[:-1]
    if not n:return np.zeros((*shape,4),np.float32)
    index={key:i for i,key in enumerate(ids)}
    xy=np.array([np.asarray(nodes[key]['position'])[[0,2]] for key in ids]);a=np.eye(n)
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for e in graph['global_edges']:
        if e['source'] not in index or e['target'] not in index:raise ValueError('invalid observed edge endpoint')
        i,j=index[e['source']],index[e['target']];a[i,j]=a[j,i]=1;parent[root(i)]=root(j)
    degree=a.sum(-1)-1;transition=a/a.sum(-1,keepdims=True)
    diffused=transition@transition@transition@xy
    component=np.array([root(i) for i in range(n)]);radius=np.zeros(n)
    for c in set(component):
        selected=component==c;radius[selected]=np.linalg.norm(xy[selected]-xy[selected].mean(0),axis=-1).max(initial=0)
    # Actual edge identities affect three-hop transport and component spread.
    node_features=np.c_[(diffused-xy)/4,radius/10,degree/8]
    flat=np.asarray(points).reshape(-1,2);distance=((flat[:,None]-xy[None])**2).sum(-1)
    nearest=np.argsort(distance,axis=-1)[:,:min(3,n)]
    weights=1/(np.take_along_axis(distance,nearest,-1)+.05);weights/=weights.sum(-1,keepdims=True)
    return (node_features[nearest]*weights[...,None]).sum(1).reshape(*shape,4).astype(np.float32)


def enrich(split):
    from .data import load_records,jobs_for
    from .features import OFFSETS
    records=load_records(split);jobs={r['root_id']:Path(folder) for folder,r in jobs_for('dev' if split.startswith('dev') else 'train')}
    grouped=collections.defaultdict(list)
    for r in records:grouped[r['root_id']].append(r)
    static=re.compile(rb'("static_updates"\s*:\s*)\[(?:\s*\{[^{}]*\}\s*,?)*\s*\]')
    start=time.perf_counter()
    for root,rows in grouped.items():
        folder=jobs[root]
        with gzip.open(folder/'legal_root.json.gz','rt') as f:graph=json.load(f)['partial_graph']
        with gzip.open(folder/'future_observations.json.gz','rb') as f:raw=f.read()
        compact,removed=static.subn(rb'\1[]',raw)
        if removed!=raw.count(b'"static_updates"'):compact=raw
        nodes=json.loads(compact)
        with gzip.open(folder/'queries.json.gz','rt') as f:queries=json.load(f)
        qi=max(range(len(queries)),key=lambda i:(len(queries[i]['compiled']['actions']),queries[i]['query_id']))
        with gzip.open(folder/'outcomes.json.gz','rt') as f:outcome=json.load(f)[qi]
        branch=[nodes[i] for i in outcome['node_ids']];previous=0
        for r in sorted(rows,key=lambda r:r['anchor_index']):
            for node in branch[previous:r['anchor_index']]:
                invalid={tuple(k) for k in node['graph_delta']['invalidated_free_cells']}
                graph=dict(graph,global_nodes={k:v for k,v in graph['global_nodes'].items() if tuple(v['cell']) not in invalid},
                           global_edges=[] if invalid else graph['global_edges'])
            previous=r['anchor_index']
            r['graph']=np.concatenate((r['graph'],topology_features(graph,r['origin'][:,None]+OFFSETS)),axis=-1)
            r['global_graph']=np.concatenate((r['global_graph'],topology_features(graph,r['robotxy'][None]+OFFSETS)),axis=-1)
    dest=OUT/'prepared'/(split+'_topology');dest.mkdir(exist_ok=False);shards=[]
    for i in range(0,len(records),1000):
        path=dest/('%05d.npy'%len(shards));np.save(path,np.asarray(records[i:i+1000],object),allow_pickle=True)
        shards.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),anchors=min(1000,len(records)-i)))
    dump(dest/'manifest.json',dict(schema='two_stage_actual_topology_enrichment_1',roots=len(grouped),anchors=len(records),shards=shards,
        parent_sha256=sha(OUT/'prepared'/split/'manifest.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-start,
        interpretation='Original10 features unchanged;4 observed-edge diffusion/component channels appended; only arrived branch invalidations applied.'))
    print(json.dumps(dict(split=split,roots=len(grouped),seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split');a=p.parse_args();enrich(a.split)
