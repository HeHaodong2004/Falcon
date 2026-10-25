"""Group queries by exact starting graph, keeping original sampling weights."""
from collections import Counter,defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info
from .diagnose import OUT,HF,PREP,read,sha,source_rows
from .shared_map import map_canvas
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment

SOURCE_ROWS=None


def ensure_times(rid,split,packed):
    global SOURCE_ROWS
    for folder in [HF/'temporal'/split,OUT/'shared_map/temporal'/split]:
        p=folder/(rid+'.json')
        if p.exists():
            tr=read(p)
            if tr['packed_sha256']!=packed['sha256']:raise ValueError('stale timestamp/graph binding')
            return tr
    if split!='full':raise FileNotFoundError('missing pilot timestamp sidecar')
    if SOURCE_ROWS is None:SOURCE_ROWS={r['root_id']:r for r in source_rows('train')[0]}
    from eventwm_graph.stage2_hf import temporal_data as td
    dest=OUT/'shared_map/temporal'/split;dest.mkdir(parents=True,exist_ok=True);lock=dest/(rid+'.lock')
    try:lock.mkdir()
    except FileExistsError:
        while not (dest/(rid+'.json')).exists():time.sleep(.1)
        return read(dest/(rid+'.json'))
    try:
        td.DEST=dest;td.SPLIT=split;return td.root_times_fast(SOURCE_ROWS[rid])
    finally:lock.rmdir()


class SharedGraphs(IterableDataset):
    def __init__(self,m,split,epoch):
        self.m=m;self.split=split;self.epoch=epoch
        roots={e['root_id']:e['scene_id']for s in m['shards']for e in s['segments']};counts=Counter(roots.values())
        self.scene_factor={s:len(roots)/(len(counts)*n)for s,n in counts.items()}
        self.n_groups=sum(len({(e['root_id'],e['start_node'])for e in s['segments']})for s in m['shards'])
    def __iter__(self):
        torch.set_num_threads(1);info=get_worker_info();wi=info.id if info else 0;nw=info.num_workers if info else 1;rng=np.random.default_rng(3011+self.epoch)
        order=rng.permutation(len(self.m['shards']))[wi::nw]
        for si in order:
            shard=self.m['shards'][int(si)]
            with np.load(shard['path'],allow_pickle=False)as z:xx=z['x'];yy=z['y'];ww=z['weight']
            byroot=defaultdict(list)
            for e in shard['segments']:byroot[e['root_id']].append(e)
            for rid in rng.permutation(list(byroot)):
                row=read(HF/'packed'/self.split/(rid+'.json'));a=load_arrays(row);tr=ensure_times(rid,self.split,row)
                with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
                lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(row['segments'])};group=defaultdict(list)
                for e in byroot[rid]:group[e['start_node']].append(e)
                for sid in rng.permutation(list(group)):
                    entries=group[sid];v=get_segment(a,lookup[(sid,entries[0]['end_node'])]);gi=int(a['graph_index'][lookup[(sid,entries[0]['end_node'])]]);gs=slice(*a['goffsets'][gi:gi+2]);p=v[3]
                    canvas=map_canvas(v[0],v[1],v[2],last[gs],p[0,0],v[-1],p[0,1:4]);xs=[];ys=[];weights=[];ij=[];cell_weights=[];window_ids=[];window_weights=[];eps=[]
                    for j,e in enumerate(entries):
                        vv=get_segment(a,lookup[(sid,e['end_node'])]);path=vv[3];sl=slice(e['start'],e['end']);x=xx[sl,16:].astype(np.float32);y=yy[sl];w=ww[sl].astype(np.float32)
                        centers=x[:,20:22].astype(np.float64)*10+path[0,[1,3]];keys=np.floor(centers/v[-1][0]).astype(np.int64)
                        truth={tuple(k):int(s)for k,s in zip(vv[4],vv[5])}
                        if not np.array_equal(y,[truth.get(tuple(k),0)for k in keys]):raise ValueError('sample key/label mismatch')
                        xs.append(x);ys.append(y);weights.append(w*self.scene_factor[e['scene_id']]);ij.append(keys-canvas['lo']);cell_weights.append(w*e['segments_in_root']);window_ids.append(np.full(len(y),j,np.int64));window_weights.append(self.scene_factor[e['scene_id']]/e['segments_in_root']);eps.append(1/max(e['candidate_cells'],1))
                    yield dict(maps=canvas['maps'],x=np.concatenate(xs),y=np.concatenate(ys),weight=np.concatenate(weights),indices=np.concatenate(ij),cell_weight=np.concatenate(cell_weights),window_id=np.concatenate(window_ids),window_weight=np.array(window_weights,np.float32),eps=np.array(eps,np.float32))


def collate(items):
    n=len(items);h=max(d['maps'].shape[1]for d in items);w=max(d['maps'].shape[2]for d in items);maps=np.zeros((n,8,h,w),np.float32);sizes=[];values=defaultdict(list);windows=0
    for i,d in enumerate(items):
        a,b=d['maps'].shape[1:];maps[i,:,:a,:b]=d['maps'];sizes.append([a,b])
        for k in ['x','y','weight','indices','cell_weight','window_weight','eps']:values[k].append(d[k])
        values['window_id'].append(d['window_id']+windows);windows+=len(d['window_weight']);values['map_ids'].append(np.full(len(d['y']),i,np.int64))
    out={k:np.concatenate(v)for k,v in values.items()};out.update(maps=maps,sizes=np.array(sizes,np.int64))
    return {k:torch.from_numpy(v)for k,v in out.items()}
