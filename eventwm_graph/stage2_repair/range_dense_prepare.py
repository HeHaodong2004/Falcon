"""Full-cell legal features for fresh, unbiased training sampling each epoch.

The capture predictor receives no targets. Labels are attached AFTER feature
extraction and every original candidate remains in the persisted universe.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_api import RangeEvidencePredictor
from .range_history import ancestor_frames
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay


class Capture(RangeEvidencePredictor):
    def _network(self,xs,details):
        extra=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation);self.captured=[];out=[]
        for x,e in zip(xs,extra):
            self.captured.append((x.cpu().numpy(),e.cpu().numpy()));v=np.zeros((len(x),4),np.float32);v[:,0]=1.;out.append(v)
        return out


def prepare(roots=64,shard=0,shards=1,device='cuda:0'):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;raw,source=source_rows('train');eligible={r['root_id']:r for r in read(HF/'packed/pilot_fit/manifest.json')['rows']};selected=[r for r in raw if r['root_id']in eligible][:roots][shard::shards];dest=OUT/'range_dense_train';dest.mkdir(exist_ok=True);model=Capture(OUT/'training/range_evidence64/selected.pt',device=device);records=[];tick=time.perf_counter()
    sources={n:sha(Path(__file__).with_name(n))for n in ['range_dense_prepare.py','range_api.py','range_fast.py','range_gpu.py','global_features.py','features.py']}
    for row in selected:
        rid=row['root_id'];path=dest/(rid+'.npz');receipt=path.with_suffix('.json');sp=OUT/'train_range_branches'/(rid+'.npz');identity=dict(schema='legal_range_full_cells_1',packed_sha256=eligible[rid]['sha256'],sensor_sha256=sha(sp),sources=sources)
        if receipt.exists():
            old=read(receipt)
            if old['identity']!=identity or old['sha256']!=sha(path):raise ValueError('dense legal features invalidated')
            records.append(old);continue
        legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        with np.load(sp)as z:sensors={k:z[k]for k in z.files}
        bystart=defaultdict(list)
        for ex in replay(row,'mixed'):
            if'audit'not in ex:bystart[ex['start_node']].append(ex)
        arrays=defaultdict(list);segments=[];cursor=0
        for sid,examples in bystart.items():
            take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
            for begin in range(0,len(examples),6):
                batch=examples[begin:begin+6];out=model.predict_segments(partial_graph=batch[0]['graph'],trajectories=[e['trajectory']for e in batch],range_observations=packet,track_history=[tracks[j]for j in take],materialize=False)
                for ex,delta,(x,extra)in zip(batch,out,model.captured):
                    keys=delta['candidate_cells'];y,_,missing=labels(keys,ex['target']);assert np.count_nonzero(y)+len(missing)==len(ex['target']);arrays['x'].append(x);arrays['extra'].append(extra);arrays['y'].append(y);arrays['keys'].append(keys);p=ex['trajectory'];segments.append(dict(start_node=sid,end_node=ex['end_node'],start=cursor,end=cursor+len(y),out_of_domain=len(missing),poses=len(p)));cursor+=len(y)
        temp=path.with_suffix('.tmp.npz');np.savez(temp,**{k:np.concatenate(v)for k,v in arrays.items()});temp.replace(path);record=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,segments=segments,rows=cursor);save(receipt,record);records.append(record);print('ROOT',len(records),rid,'rows',cursor,'seconds',time.perf_counter()-tick,flush=True)
    name='manifest.json'if shards==1 else'manifest_part%d.json'%shard;save(dest/name,dict(schema='legal_range_full_cells_1',roots=len(records),scenes=len({r['scene_id']for r in records}),records=records,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=64);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--device',default='cuda:0');prepare(**vars(p.parse_args()))
