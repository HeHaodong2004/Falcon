"""Current-map training with fresh stratified cell draws each epoch.

All original windows remain. Negatives near known-map frontiers receive a
separate sampling stratum with exact inverse inclusion weights. The rest of
the candidate domain remains sampled and evaluated, with no hard pruning.
"""
import argparse
from collections import Counter,OrderedDict,defaultdict
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from .diagnose import OUT,HF,CKPT,read,save,sha
from .shared_map import SharedMapNet,map_canvas
from .shared_data import ensure_times
from .shared_train import loss_fn
from eventwm_graph.stage2_hf.common import source
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from eventwm_graph.stage2_segment.batch import SharedCompiledGraph
from eventwm_graph.stage2_segment.features import SegmentFeatures
from eventwm_graph.stage2_segment.cuda import CudaExecution
from .features import transform_batch
from .full_evaluate import labels


class OnlineGraphs(IterableDataset):
    def __init__(self,m,split,epoch,anchored_fine=False,graphs_per_root=0):
        self.m=m;self.split=split;self.epoch=epoch;self.anchored_fine=anchored_fine;self.graphs_per_root=graphs_per_root;counts=Counter(r['scene_id']for r in m['rows']);self.scene_factor={s:m['roots']/(len(counts)*n)for s,n in counts.items()}
        group_counts=[len({e['start_node']for e in r['segments']})for r in m['rows']]
        self.eligible_groups=sum(group_counts)
        self.n_groups=sum(min(n,graphs_per_root)if graphs_per_root else n for n in group_counts)
    def __iter__(self):
        torch.set_num_threads(1);info=get_worker_info();wi=info.id if info else 0;nw=info.num_workers if info else 1;rng=np.random.default_rng(3011+self.epoch)
        for ri in rng.permutation(len(self.m['rows']))[wi::nw]:
            row=self.m['rows'][int(ri)];a=load_arrays(row);rid=row['root_id'];tr=ensure_times(rid,self.split,row)
            with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
            group=defaultdict(list)
            for i,e in enumerate(row['segments']):group[e['start_node']].append(i)
            chosen=rng.permutation(list(group))
            if self.graphs_per_root:chosen=chosen[:self.graphs_per_root]
            inclusion_weight=len(group)/len(chosen)
            for sid in chosen:
                ids=group[sid];v=get_segment(a,ids[0]);gi=int(a['graph_index'][ids[0]]);gs=slice(*a['goffsets'][gi:gi+2]);p=v[3]
                d=map_canvas(v[0],v[1],v[2],last[gs],p[0,0],v[-1],p[0,1:4],anchor=self.anchored_fine,halo=128 if self.anchored_fine else 96);windows=[]
                for i in ids:
                    vv=get_segment(a,i);windows.append(dict(path=vv[3],target_keys=vv[4],target_states=vv[5],end_node=row['segments'][i]['end_node']))
                yield dict(maps=d['maps'],fine=d['fine']if self.anchored_fine else None,lo=d['lo'],keys=v[0],states=v[1],heights=v[2],geometry=v[-1],now=float(p[0,0]),windows=windows,scene_factor=self.scene_factor[row['scene_id']]*inclusion_weight,windows_in_root=len(row['segments']),root_id=rid)


class ArrayPrepared(SharedCompiledGraph):
    def __init__(self,d):
        self.resolution=float(d['geometry'][0]);self.height=float(d['geometry'][1]);self.keys=d['keys'];self.values=d['states'];self.heights=d['heights'];self.height64=self.heights.astype(np.float64)
        self.crops=OrderedDict();self.crop_hits=0;self.crop_misses=0;self.integrals=None;self.shared=None;self.shared_builds=0;self.shared_slices=0;self.shared_height_builds=0;self.crop_cache_size=8;self.device_cache_size=8
        self.known_lo=self.keys.astype(np.int64).min(0)if len(self.keys)else None;self.known_hi=self.keys.astype(np.int64).max(0)+1 if len(self.keys)else None
        self.uniform=bool(len(self.heights)and np.all(self.heights.view(np.uint32)==self.heights[:1].view(np.uint32)))


def keep(items):return items


def make_batch(items,engine,rng,device):
    n=len(items);h=max(d['maps'].shape[1]for d in items);w=max(d['maps'].shape[2]for d in items);maps=np.zeros((n,items[0]['maps'].shape[0],h,w),np.float32);fine=np.zeros((n,5,h*2,w*2),np.float32)if items[0].get('fine')is not None else None;sizes=[];xs=[];indices=[];mapids=[];ys=[];weights=[];cellweights=[];windowids=[];windowweights=[];eps=[];wid=0
    for mi,d in enumerate(items):
        a,b=d['maps'].shape[1:];maps[mi,:,:a,:b]=d['maps'];sizes.append([a,b]);prepared=ArrayPrepared(d);paths=[e['path']for e in d['windows']];prepared.prewarm(paths)
        if fine is not None:fine[mi,:,:a*2,:b*2]=d['fine']
        graph=dict(schema='causal_partial_cspace_graph_1',timestamp=d['now'],resolution=prepared.resolution,height=prepared.height,clearance=float(d['geometry'][2]));contexts=[];keysets=[];selected_meta=[]
        for e in d['windows']:
            ctx=SegmentFeatures(graph,e['path'],prepared);keys=ctx.candidates('both');target={tuple(k):(int(s),0.)for k,s in zip(e['target_keys'],e['target_states'])};y,_,missing=labels(keys,target)
            if missing:raise ValueError('training positive outside original candidate universe')
            ij=keys-ctx.lo;near=np.minimum(ctx.dist[0][ij[:,0],ij[:,1]],ctx.dist[1][ij[:,0],ij[:,1]])<=1.
            select=[];ww=[]
            for mask,maximum in [(y>0,128),((y==0)&near,256),((y==0)&~near,128)]:
                ids=np.flatnonzero(mask);count=min(maximum,len(ids))
                if count:select.extend(rng.choice(ids,count,replace=False));ww.extend([len(ids)/(count*max(len(keys),1))]*count)
            select=np.asarray(select,np.int64);keysets.append(keys[select]);contexts.append(ctx);selected_meta.append((y[select],np.asarray(ww,np.float32),len(keys)))
        xlist,_,_=transform_batch(engine,contexts,keysets)
        for x,keys,(y,cw,ncells)in zip(xlist,keysets,selected_meta):
            xs.append(x);indices.append(keys-d['lo']);mapids.append(np.full(len(y),mi,np.int64));ys.append(y);cellweights.append(cw);weights.append(cw*d['scene_factor']/d['windows_in_root']);windowids.append(np.full(len(y),wid,np.int64));windowweights.append(d['scene_factor']/d['windows_in_root']);eps.append(1/max(ncells,1));wid+=1
    b=dict(maps=torch.from_numpy(maps).to(device),sizes=torch.tensor(sizes,device=device),x=torch.cat(xs))
    if fine is not None:b['fine']=torch.from_numpy(fine).to(device)
    for name,values in [('indices',indices),('map_ids',mapids),('y',ys),('weight',weights),('cell_weight',cellweights),('window_id',windowids)]:b[name]=torch.from_numpy(np.concatenate(values)).to(device)
    b['window_weight']=torch.tensor(windowweights,device=device);b['eps']=torch.tensor(eps,device=device);return b


def fit(name='shared_online_pilot',split='pilot_fit',epochs=2,device='cuda:0',workers=6,batch_size=8,dice=.2,warm=None,anchored_fine=False,graphs_per_root=0):
    torch.set_num_threads(1);random.seed(3011);np.random.seed(3011);torch.manual_seed(3011);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if graphs_per_root<0:raise ValueError('negative graph sampling count')
    path=HF/'packed'/split/'manifest.json';m=read(path);data=OnlineGraphs(m,split,0,anchored_fine,graphs_per_root);dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False);model_config=dict(map_channels=10 if anchored_fine else 8,fine=anchored_fine);model=SharedMapNet(**model_config);point=OUT/'training/point_residual_full/epoch06.pt';model.initialize(point,HF/'training/spatial_bce/epoch08.pt')
    if warm:model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
    model.to(device);engine=CudaExecution(model.point.prior,device);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();rng=np.random.default_rng(3011);scale=data.n_groups/m['roots'];tick=time.perf_counter();logs=[]
    cfg=dict(schema='stage2_shared_map_1',name=name,split=split,epochs=epochs,seed=3011,source=source(),sources={p.name:sha(p)for p in Path(__file__).parent.glob('shared_*.py')},point_checkpoint=str(point),point_checkpoint_sha256=sha(point),manifest_sha256=sha(path),roots=m['roots'],windows=m['segments'],starting_graphs=data.n_groups,learning_rate=.0005,batch_size=batch_size,workers=workers,pos_weight=8.,dice_weight=dice,warm=warm,
        sampling='Fresh128positive/256UNKNOWNwithin1mknown/128remainingUNKNOWN perwindow perpass; exactinverseinclusion; full original candidates evaluated. Sampling coordinates only gather loss, not supplied as a mask to encoder.',
        model_config=model_config,inputs='Shared current legal map states/heights/EDTs/certificate ages; optional current-ego XY anchor and original-resolution static map branch; original88legal map+suppliedtrajectory covariates from exact CUDA kernel. No future/camera/human truth inputs.',weighting='EqualScene/Root/window with exact cell-stratum inverse inclusion; grouped starting graphs only reuse compute.')
    save(dest/'config.json',cfg)
    cfg.update(graphs_per_root=graphs_per_root,eligible_starting_graphs=data.eligible_groups,graph_sampling='All eligible training Roots every pass; if bounded, uniform starting-graph sampling without replacement with inverse inclusion len(allgraphs)/len(selectedgraphs). Every selected graph retains all its original windows. Full original universe remains eligible and development is exhaustive.')
    save(dest/'config.json',cfg)
    for epoch in range(1,epochs+1):
        loader=DataLoader(OnlineGraphs(m,split,epoch,anchored_fine,graphs_per_root),batch_size=batch_size,num_workers=workers,collate_fn=keep,prefetch_factor=2);steps=0;ng=0;nw=0;visited=set()
        for items in loader:
            visited.update(d['root_id']for d in items)
            b=make_batch(items,engine,rng,device);optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():encoded=model.encode(b['maps'],b['sizes'],b.get('fine'))
            r,bl,_=model.raw(b['x'],b['indices'],b['map_ids'],encoded);loss,parts=loss_fn(r,bl,b,scale,dice)
            if not torch.isfinite(loss):raise ValueError('nonfinite shared online loss')
            scaler.scale(loss).backward();scaler.unscale_(optim);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optim);scaler.update();steps+=1;ng+=len(items);nw+=len(b['window_weight'])
            if steps%50==0:print(dict(epoch=epoch,steps=steps,graphs=ng,windows=nw,seconds=time.perf_counter()-tick,loss=float(loss),gradient_norm=float(grad),map_gradient=float(model.enc0.c1.weight.grad.norm())),flush=True)
        assert ng==data.n_groups and len(visited)==m['roots']
        if not graphs_per_root:assert nw==m['segments']
        logs.append(dict(epoch=epoch,steps=steps,roots=len(visited),graphs=ng,windows=nw,eligible_windows=m['segments'],seconds=time.perf_counter()-tick));save(dest/'learning.json',logs);cp=dest/('epoch%02d.pt'%epoch)
        torch.save(dict(schema=cfg['schema'],config=cfg,model_config=model_config,model=model.state_dict(),epoch=epoch,threshold=.2),cp)
        from .shared_eval import evaluate
        evaluate(cp,device=device,name=name+'_epoch%02d'%epoch)
    save(dest/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-tick,parameters=sum(p.numel()for p in model.parameters()),peak_cuda_allocated=torch.cuda.max_memory_allocated(device)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='shared_online_pilot');p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=2);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=6);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--dice',type=float,default=.2);p.add_argument('--warm');p.add_argument('--anchored-fine',action='store_true');p.add_argument('--graphs-per-root',type=int,default=0);fit(**vars(p.parse_args()))
