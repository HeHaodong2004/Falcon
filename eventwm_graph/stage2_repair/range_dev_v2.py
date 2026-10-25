"""Capture current public API features on the original fixed pilot.

Targets are attached after feature extraction. Preserve the original row/key
ordering, reuse byte-identical88feature arrays, and record exact differences.
"""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_radix_api import RadixExactSpatialPredictor
from .range_history import ancestor_frames
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay


class Capture(RadixExactSpatialPredictor):
    def _network(self,xs,details):
        extras=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation)
        self.features=[(x.cpu().numpy(),e.cpu().numpy())for x,e in zip(xs,extras)]
        out=[]
        for x in xs:
            values=np.zeros((len(x),4),np.float32);values[:,0]=1.;out.append(values)
        return out


def run(device='cuda:1'):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter()
    old=read(OUT/'range_evidence_dev/manifest.json');raw,_=source_rows('dev');raw={r['root_id']:r for r in raw}
    dest=OUT/'range_evidence_dev_v2';dest.mkdir(exist_ok=True)
    model=Capture(OUT/'training/support_pretrained256_control/selected.pt',device=device)
    model.config['observed_obstacle_schema']='capped_2m_complete';records=[]
    sources={name:sha(Path(__file__).with_name(name))for name in ['range_dev_v2.py','range_api.py','global_features.py','features.py','range_fast.py','range_gpu.py','range_planar.py','radix_bvh.py']}
    for row in old['records']:
        rid=row['root_id'];path=dest/(rid+'.npz');receipt=path.with_suffix('.json');sp=OUT/'range_branches'/(rid+'.npz')
        identity=dict(schema='legal_public_features88_evidence16_v2',observed_obstacle_schema='capped_2m_complete',original_evidence_sha256=row['sha256'],original_features_sha256=sha(row['original_feature_path']),sensor_sha256=sha(sp),sources=sources)
        if receipt.exists():
            prior=read(receipt)
            if prior['identity']!=identity or prior['sha256']!=sha(path):raise ValueError('public feature cache source changed')
            records.append(prior);continue
        with np.load(row['original_feature_path'])as z:oldx=z['x'];oldy=z['y']
        with np.load(row['path'])as z:oldextra=z['extra'];np.testing.assert_array_equal(oldy,z['y'])
        x=np.empty_like(oldx);extra=np.empty_like(oldextra);keys=np.empty((len(x),2),np.int32);segments={(e['start_node'],e['end_node']):e for e in row['segments']}
        legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        with np.load(sp)as z:sensors={k:z[k]for k in z.files}
        grouped=defaultdict(list)
        for ex in replay(raw[rid],'mixed'):
            if 'audit'not in ex:grouped[ex['start_node']].append(ex)
        for sid,examples in grouped.items():
            take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
            outputs=model.predict_segments(partial_graph=examples[0]['graph'],trajectories=[e['trajectory']for e in examples],range_observations=packet,track_history=[tracks[j]for j in take],materialize=False)
            for ex,delta,(xx,ee)in zip(examples,outputs,model.features):
                e=segments[ex['start_node'],ex['end_node']];sl=slice(e['start'],e['end']);kk=delta['candidate_cells'];p=ex['trajectory']
                expected=np.floor((oldx[sl,20:22].astype(np.float64)*10+p[0,[1,3]])/ex['graph']['resolution']).astype(np.int32)
                np.testing.assert_array_equal(kk,expected);y,_,missing=labels(kk,ex['target']);np.testing.assert_array_equal(y,oldy[sl])
                assert len(xx)==sl.stop-sl.start
                x[sl]=xx;extra[sl]=ee;keys[sl]=kk
        difference_x=np.abs(x-oldx);difference_extra=np.abs(extra-oldextra)
        unchanged=bool(np.array_equal(x,oldx));arrays=dict(extra=extra,y=oldy,keys=keys)
        if not unchanged:arrays['x']=x
        np.savez(path,**arrays)
        record=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,original_feature_path=row['original_feature_path'],segments=row['segments'],rows=len(x),reuses_original_x88=unchanged,x88_changed_rows=int(np.any(x!=oldx,1).sum()),x88_max_difference_by_column=difference_x.max(0).tolist(),extra16_changed_rows=int(np.any(extra!=oldextra,1).sum()),extra16_max_difference_by_column=difference_extra.max(0).tolist())
        save(receipt,record);records.append(record);model.clear_cache()
        print('ROOT',len(records),rid,'x_changed',record['x88_changed_rows'],'extra_changed',record['extra16_changed_rows'],'seconds',time.perf_counter()-tick,flush=True)
    save(dest/'manifest.json',dict(schema='legal_public_features88_evidence16_v2',role='Original666window pilot inputs captured from actual public API. Future target attached only afterward. Old caches remain unchanged.',observed_obstacle_schema='capped_2m_complete',records=records,roots=len(records),windows=sum(len(r['segments'])for r in records),seconds=time.perf_counter()-tick))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');run(**vars(p.parse_args()))
