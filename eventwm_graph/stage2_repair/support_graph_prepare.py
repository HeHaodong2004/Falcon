"""Legal partial-graph canvases and SEPARATE training-only support targets.

Expands encoder supervision without needing new sensor collection. Range
channels are explicitly absent/zero in this pretraining set. No scene ID,
navigation data, or future certificate enters the encoder input arrays.
"""
import argparse
from multiprocessing import get_context
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_bev import SIZE
from .support_teacher import navigation_targets,crop_targets


def legal_canvas(keys,states,heights,ego,resolution):
    lo=(np.floor(np.asarray(ego)[[0,2]]/resolution).astype(np.int64)//32)*32-SIZE
    n=2*SIZE;grid=np.zeros((n,n),np.uint8);height=np.zeros((n,n),np.float32)
    ij=keys-lo;valid=np.all((ij>=0)&(ij<n),1);ij=ij[valid]
    grid[ij[:,0],ij[:,1]]=states[valid];height[ij[:,0],ij[:,1]]=heights[valid]-ego[1]
    pool=lambda a:a.reshape(SIZE,2,SIZE,2).mean((1,3))
    free=pool((grid==1).astype(np.float32));blocked=pool((grid==2).astype(np.float32));known=free+blocked
    distances=[pool(distance_transform_edt(grid!=s)*resolution/10)if np.any(grid==s)else np.full_like(free,1.1)for s in [1,2]]
    graph=np.stack([free,blocked,pool(height)/np.maximum(known,.25),*distances,known])
    xy=(np.arange(SIZE)*2+1)*resolution
    anchors=np.stack([np.broadcast_to((xy+lo[0]*resolution-ego[0])[:,None]/10,(SIZE,SIZE)),np.broadcast_to((xy+lo[1]*resolution-ego[2])[None,:]/10,(SIZE,SIZE))])
    return np.concatenate([graph,np.zeros((6,SIZE,SIZE),np.float32),anchors]).astype(np.float16),lo


def one(task):
    packed,source=task;rid=packed['root_id'];dest=OUT/'support_graph_train';path=dest/(rid+'.npz');receipt=path.with_suffix('.json')
    identity=dict(schema='training_legal_graph_support_pretrain_1',packed_sha256=packed['sha256'],producer_sha256=sha(__file__),teacher_sha256=sha(Path(__file__).with_name('support_teacher.py')))
    if receipt.exists():
        old=read(receipt)
        if old['identity']!=identity or old['sha256']!=sha(path):raise ValueError('stale graph-pretraining data')
        return old
    private=member(source,'privileged_initialization.json.gz')
    from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
    a=load_arrays(packed);r=a['geometry'][0]
    navkeys,navheights,navreceipt=navigation_targets(private['episode']['info']['graph_navmesh'],r)
    maps=[];labels=[];seen=set()
    for index,e in enumerate(packed['segments']):
        gi=e['graph_index']
        if gi in seen:continue
        seen.add(gi);keys,states,heights,trajectory,*_=get_segment(a,index);ego=trajectory[0,1:4]
        canvas,lo=legal_canvas(keys,states,heights,ego,r)
        target=crop_targets(navkeys,navheights,lo,ego[1],r)
        maps.append(canvas);labels.append(target)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,maps=np.array(maps),support_targets=np.array(labels));temp.replace(path)
    result=dict(root_id=rid,scene_id=packed['scene_id'],path=str(path),sha256=sha(path),identity=identity,maps=len(maps),navigation_target_sha256=navreceipt['sha256'],inputs='Only current legal graph; six absent range channels zero; relative cell anchors. Private support is separate loss target only.')
    save(receipt,result);return result


def prepare(roots=2048,workers=4):
    tick=time.perf_counter();m=read(HF/'packed/pilot_fit/manifest.json');rows=m['rows'][:roots]
    train,_=source_rows('train');source={r['root_id']:r for r in train};dev,_=source_rows('dev')
    assert {r['root_id']for r in rows}<=source.keys()
    assert not ({r['scene_id']for r in rows}&{r['scene_id']for r in dev})
    dest=OUT/'support_graph_train';dest.mkdir(exist_ok=True);records=[]
    with get_context('spawn').Pool(workers)as pool:
        for record in pool.imap_unordered(one,[(row,source[row['root_id']])for row in rows]):
            records.append(record)
            if len(records)%32==0 or len(records)==len(rows):
                save(dest/'progress.json',dict(roots=len(records),requested=len(rows),maps=sum(r['maps']for r in records),seconds=time.perf_counter()-tick))
                print('ROOTS',len(records),'maps',sum(r['maps']for r in records),'seconds',time.perf_counter()-tick,flush=True)
    records.sort(key=lambda r:r['root_id'])
    save(dest/('manifest_roots%d.json'%len(records)),dict(schema='training_legal_graph_support_pretrain_1',roots=len(records),scenes=len({r['scene_id']for r in records}),maps=sum(r['maps']for r in records),records=records,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=2048);p.add_argument('--workers',type=int,default=4);prepare(**vars(p.parse_args()))
