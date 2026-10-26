"""Full-cell reconstruction on fixed32 TRAINING Roots, not generalization.

Measure the remaining training/development gap before changing capacity.
Each forecast gets only ancestor/current sensor frames and supplied paths.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,read,save,sha,member
from .native_triangle_api import NativeTrianglePredictor
from .range_history import ancestor_frames
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.prepared import content_key
from scripts.stage2_diagnostic.common import extended_rates


def run(device='cuda:1',roots=32,pixels=256):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter();snapshot=OUT/'training/online_sensor256_roots256/snapshot.json';rows=read(snapshot)['records'][:roots];checkpoint=OUT/'training/range_online16384_streamed/selected_epoch01_argmax.pt';model=NativeTrianglePredictor(checkpoint,device=device);records=[]
    for row in rows:
        raw=row['source'];rid=raw['root_id'];directory='expanded_train_range_branches_256'if pixels==256 else'expanded_train_range_branches';sp=OUT/directory/(rid+'.npz');receipt=read(sp.with_suffix('.json'))
        if receipt['sha256']!=sha(sp)or receipt['identity']['files']!=raw['files']:raise ValueError('training sensor source mismatch')
        with np.load(sp)as z:sensors={k:z[k]for k in z.files}
        if sensors['depth'].shape[1:]!=(pixels,pixels):raise ValueError('sensor resolution mismatch')
        legal=member(raw,'legal_root.json.gz');nodes=member(raw,'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes];groups=defaultdict(list)
        for e in replay(raw,'mixed'):
            if'audit'not in e:groups[e['start_node']].append(e)
        windows=[]
        for sid,examples in groups.items():
            take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1');graph=examples[0]['graph'];before=content_key(graph)
            predictions=model.predict_segments(partial_graph=graph,trajectories=[e['trajectory']for e in examples],range_observations=packet,track_history=[tracks[i]for i in take])
            for ex,d in zip(examples,predictions):
                y,_,missing=labels(d['candidate_cells'],ex['target']);cm=np.bincount(3*y.astype(np.int64)+d['states'],minlength=9).reshape(3,3)
                for state,height in missing:cm[state,0]+=1
                windows.append(dict(start_node=sid,end_node=ex['end_node'],confusion=cm.tolist(),target=len(ex['target']),candidates=len(y),out_of_domain=len(missing)))
            assert content_key(graph)==before
        records.append(dict(root_id=rid,scene_id=raw['scene_id'],windows=windows));model.clear_cache();print('TRAIN_ROOT',len(records),'windows',sum(len(r['windows'])for r in records),'seconds',time.perf_counter()-tick,flush=True)
    scenes=defaultdict(list);pooled=np.zeros((3,3),np.int64)
    for row in records:
        c=np.array([w['confusion']for w in row['windows']]);scenes[row['scene_id']].append(c.mean(0));pooled+=c.sum(0)
    mean=np.mean([np.mean(v,0)for v in scenes.values()],0);metrics=extended_rates(mean)
    # Independent direct set-count formulas, with the original aggregation.
    correct=mean[1,1]+mean[2,2];actual=mean[1:,:].sum();predicted=mean[:,1:].sum();intersection=mean[1:,1:].sum();assert abs(metrics['label_aware_delta_iou']-correct/(actual+predicted-correct))<1.e-12;assert abs(metrics['revelation_iou']-intersection/(actual+predicted-intersection))<1.e-12
    result=dict(role='Fixed first32eligible TRAINING Roots and all their original windows/cells; not development, confirmation or full-training acceptance. Diagnostic training gap only.',checkpoint_sha256=sha(checkpoint),snapshot_sha256=sha(snapshot),sensor_pixels=pixels,roots=len(records),scenes=len(scenes),windows=sum(len(r['windows'])for r in records),hierarchical=metrics,pooled=extended_rates(pooled),pooled_confusion=pooled.tolist(),records=records,seconds=time.perf_counter()-tick);save(OUT/('train_reconstruction_probe_sensor%d.json'%pixels),result);print('TRAIN_SUMMARY',metrics['label_aware_delta_iou'],metrics['revelation_iou'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--roots',type=int,default=32);p.add_argument('--pixels',type=int,choices=[128,256],default=256);run(**vars(p.parse_args()))
