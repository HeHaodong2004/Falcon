"""Separate checksummed starting-render sidecar; existing16feature cache retained."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import time
import numpy as np
import torch
from .diagnose import OUT,read,save,sha,source_rows,member
from .device_pipeline import ColumnPreparedPipelinePredictor
from .initial_visibility import OptixInitialKernel
from .range_history import ancestor_frames
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay


class Capture(ColumnPreparedPipelinePredictor):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.range_kernel=OptixInitialKernel(self.device,primitive_mode='triangle')
    def _encoded_map(self,details):return None,np.zeros(2,np.int64)
    def _probability_tiles(self,x,extra,indices,encoded,baseline,**kwargs):
        self.codes=extra[:,16:].argmax(1).to(torch.uint8).cpu().numpy();out=torch.zeros((len(x),5),device=self.device);out[:,0]=1;out[:,4]=baseline;return out
    def _network(self,xs,details):
        extras=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation);self.codes=np.concatenate([v[:,16:].argmax(1).to(torch.uint8).cpu().numpy()for v in extras]);out=[]
        for x in xs:
            v=np.zeros((len(x),4),np.float32);v[:,0]=1;out.append(v)
        return out


def run(device='cuda:1'):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;old=read(OUT/'range_evidence_dev_v2/manifest.json');raw,_=source_rows('dev');raw={r['root_id']:r for r in raw};dest=OUT/'range_initial_visibility_dev_v1';dest.mkdir(exist_ok=True);model=Capture(OUT/'training/range_online4096/selected_epoch02_argmax.pt',device=device);records=[];tick=time.perf_counter();sources={n:sha(Path(__file__).with_name(n))for n in ['initial_cache.py','initial_visibility.py','range_fast.py','device_range_features.py','device_query.py']}
    for row in old['records']:
        rid=row['root_id'];path=dest/(rid+'.npz');receipt=path.with_suffix('.json');sensor_path=OUT/'range_branches'/(rid+'.npz');identity=dict(schema='current_partial_mesh_initial_code_1',original_feature_sha256=row['sha256'],sensor_sha256=sha(sensor_path),source_files=raw[rid]['files'],sources=sources)
        if receipt.exists():
            prior=read(receipt)
            if prior['identity']!=identity or prior['sha256']!=sha(path):raise ValueError('initial sidecar source changed')
            records.append(prior);continue
        with np.load(row['path'])as z:oldkeys=z['keys'];oldy=z['y']
        codes=np.empty(len(oldy),np.uint8);entries={(e['start_node'],e['end_node']):dict(e)for e in row['segments']};written=set();legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        with np.load(sensor_path)as z:sensors={k:z[k]for k in z.files}
        groups=defaultdict(list)
        for ex in replay(raw[rid],'mixed'):
            if'audit'not in ex:groups[ex['start_node']].append(ex)
        for sid,examples in groups.items():
            take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1');outputs=model.predict_segments(partial_graph=examples[0]['graph'],trajectories=[e['trajectory']for e in examples],range_observations=packet,track_history=[tracks[j]for j in take],materialize=False);offset=0
            for ex,delta in zip(examples,outputs):
                key=(sid,ex['end_node']);assert key not in written;written.add(key);e=entries[key];sl=slice(e['start'],e['end']);keys=delta['candidate_cells'];np.testing.assert_array_equal(keys,oldkeys[sl]);y,_,missing=labels(keys,ex['target']);np.testing.assert_array_equal(y,oldy[sl]);codes[sl]=model.codes[offset:offset+len(keys)];offset+=len(keys);e['domain_sha256']=hashlib.sha256(keys.astype('<i4').tobytes()).hexdigest();e['out_of_domain_positives']=len(missing)
        assert written==set(entries);np.savez(path,initial_code=codes);record=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,segments=list(entries.values()),rows=len(codes));save(receipt,record);records.append(record);model.clear_cache();print('ROOT',len(records),'seconds',time.perf_counter()-tick,flush=True)
    save(dest/'manifest.json',dict(schema='current_partial_mesh_initial_code_1',role='Legal partial START-view render code sidecar only. Not actual certified visibility. Source/keys checked; targets used only afterward for identity validation. Existing16feature cache unchanged.',records=records,roots=len(records),windows=sum(len(r['segments'])for r in records),seconds=time.perf_counter()-tick))


def attach(groups,manifest):
    m=read(manifest)
    if m['schema']!='current_partial_mesh_initial_code_1':raise ValueError('wrong initial sidecar schema')
    rows={r['root_id']:r for r in m['records']};cache_rid=None
    for g in groups:
        if'initial_extra'in g:continue
        rid=g['root_id']
        if rid!=cache_rid:
            row=rows[rid]
            if sha(row['path'])!=row['sha256']:raise ValueError('initial sidecar bytes changed')
            with np.load(row['path'])as z:codes=z['initial_code']
            entries={(e['start_node'],e['end_node']):e for e in row['segments']};cache_rid=rid
        values=np.empty((len(g['y']),3),np.float32)
        for e in g['segments']:
            old=entries[e['start_node'],e['end_node']];sl=slice(e['start'],e['end']);keys=g['indices'][sl]+g['low']
            if hashlib.sha256(keys.astype('<i4').tobytes()).hexdigest()!=old['domain_sha256']:raise ValueError('initial sidecar coordinates/order changed')
            c=codes[old['start']:old['end']];assert len(c)==e['end']-e['start'];values[sl]=np.eye(3,dtype=np.float32)[c]
        g['initial_extra']=values
    return groups


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');run(**vars(p.parse_args()))
