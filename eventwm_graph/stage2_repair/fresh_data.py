"""Fresh cell draws with exact inverse-inclusion weights; no domain pruning."""
from collections import Counter
import multiprocessing as mp
import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info
from .diagnose import read


def draw(strata,candidates,windows,rng,caps=(128,384,128)):
    ids=[];weights=[]
    for population,cap in zip(strata,caps):
        n=min(len(population),cap)
        if not n:continue
        chosen=rng.choice(population,n,replace=False);ids.append(chosen);weights.append(np.full(n,len(population)/(n*max(candidates,1)*windows),np.float32))
    if not ids:return np.empty(0,np.int64),np.empty(0,np.float32)
    ids=np.concatenate(ids);weights=np.concatenate(weights);order=np.argsort(ids);return ids[order],weights[order]


class FreshGroups(IterableDataset):
    def __init__(self,manifest):
        self.manifest=read(manifest);self.epoch=mp.Value('i',0);self.groups=[];m=self.manifest;counts=Counter(r['scene_id']for r in m['records']);factor={s:m['roots']/(len(counts)*n)for s,n in counts.items()}
        for row in m['records']:
            with np.load(row['point_path'])as z:arrays={k:z[k]for k in ['x','extra','y','keys']}
            for a in arrays.values():a.flags.writeable=False
            with np.load(row['map_path'])as z:maps=z['maps'];low=z['lo'];aux=z['aux'];mask=z['mask']
            windows=sum(len(g['segments'])for g in row['groups']);near=(np.minimum(arrays['x'][:,18],arrays['x'][:,19])*10<=1.)
            for g in row['groups']:
                strata=[]
                for e in g['segments']:
                    sl=slice(e['start'],e['end']);y=arrays['y'][sl];n=near[sl];populations=[np.flatnonzero(y>0)+e['start'],np.flatnonzero((y==0)&n)+e['start'],np.flatnonzero((y==0)&~n)+e['start']];strata.append((e,populations))
                mi=g['map_index'];self.groups.append(dict(arrays=arrays,maps=maps[mi],low=low[mi],aux=aux[mi],mask=mask[mi],strata=strata,root_windows=windows,scene_factor=factor[row['scene_id']],aux_weight=len(g['segments'])/windows*factor[row['scene_id']],root_id=row['root_id']))

    def __iter__(self):
        torch.set_num_threads(1);info=get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1;epoch=self.epoch.value;order=np.random.default_rng(9017+epoch).permutation(len(self.groups))
        for gi in order[worker::workers]:
            g=self.groups[int(gi)];rng=np.random.default_rng(8101+epoch*1000003+int(gi));indices=[];weights=[]
            for e,populations in g['strata']:
                ids,w=draw(populations,e['end']-e['start'],g['root_windows'],rng);indices.append(ids);weights.append(w)
            ids=np.concatenate(indices);weight=np.concatenate(weights)*g['scene_factor'];a=g['arrays']
            yield dict(maps=g['maps'],aux=g['aux'],mask=g['mask'],aux_weight=g['aux_weight'],x=a['x'][ids],extra=a['extra'][ids],y=a['y'][ids],indices=a['keys'][ids]-g['low'],weight=weight)


def keep(items):return items
