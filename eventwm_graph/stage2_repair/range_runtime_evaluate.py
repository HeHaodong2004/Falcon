"""Real public API evaluation on original fixed pilot, all cells/windows."""
import argparse
from collections import defaultdict
import hashlib
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_api import RangeEvidencePredictor
from .range_history import ancestor_frames
from .full_evaluate import labels,codes
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.prepared import content_key
from scripts.stage2_diagnostic.common import extended_rates


def run(checkpoint,device='cuda:0',stride=1,planar=False,proxy=False,spatial=False,radix=False):
    cls=RangeEvidencePredictor
    if planar:
        from .range_planar_api import PlanarRangePredictor
        cls=PlanarRangePredictor
    if spatial:
        from .range_spatial_api import SpatialRangePredictor
        cls=SpatialRangePredictor
    if proxy:
        from .range_fast_proxy import FastRangePredictor,FastSpatialPredictor
        cls=FastSpatialPredictor if spatial else FastRangePredictor
    if radix:
        from .range_radix_api import RadixRangePredictor,RadixExactPredictor,RadixSpatialPredictor,RadixExactSpatialPredictor
        cls=(RadixSpatialPredictor if proxy else RadixExactSpatialPredictor)if spatial else RadixRangePredictor if proxy else RadixExactPredictor
    label='spatial_'if spatial else'';label+='proxy7'if proxy else'planar'if planar else'stride%d'%stride
    if radix:label+='_radix'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;model=cls(checkpoint,device=device,mesh_stride=stride);raw,source=source_rows('dev');raw={r['root_id']:r for r in raw};manifest=read(HF/'light_eval_cache/manifest.json');dest=OUT/('range_runtime_'+label);dest.mkdir(exist_ok=True);roots=[];tick=time.perf_counter()
    for row in manifest['rows']:
        rid=row['root_id'];legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        with np.load(OUT/'range_branches'/(rid+'.npz'))as z:sensors={k:z[k]for k in z.files}
        with np.load(row['path'])as z:oldx=z['x'];oldy=z['y'];oldprob=z['baseline']
        old={(e['start_node'],e['end_node']):e for e in row['segments']};groups=defaultdict(list);records=[]
        for ex in replay(raw[rid],'mixed'):
            if'audit'not in ex:groups[ex['start_node']].append(ex)
        for sid,examples in groups.items():
            graph=examples[0]['graph'];before=content_key(graph);take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1');paths=[e['trajectory']for e in examples]
            results=model.predict_segments(partial_graph=graph,trajectories=paths,range_observations=packet,track_history=[tracks[j]for j in take])
            for ex,d in zip(examples,results):
                e=old[ex['start_node'],ex['end_node']];sl=slice(e['start'],e['end']);keys=d['candidate_cells'];expected=np.floor((oldx[sl,20:22].astype(np.float64)*10+ex['trajectory'][0,[1,3]])/graph['resolution']).astype(np.int32);np.testing.assert_array_equal(keys,expected);y,heights,missing=labels(keys,ex['target']);np.testing.assert_array_equal(y,oldy[sl]);cm=np.bincount(3*y.astype(np.int64)+d['states'],minlength=9).reshape(3,3)
                for state,height in missing:cm[state,0]+=1
                bp=oldprob[sl];bh=np.where(bp[:,1]+bp[:,2]>=.2,np.where(bp[:,1]>=bp[:,2],1,2),0);bc=np.bincount(3*y.astype(np.int64)+bh,minlength=9).reshape(3,3)
                future=d['future_partial_graph'];columns=future.cell_columns;prepared=model.prepare_graph(graph);base=prepared.future_base.table;fc=codes(columns['cell']);ki=np.searchsorted(fc,codes(base.keys));np.testing.assert_array_equal(columns['cell'][ki],base.keys)
                for name,value in [('state',base.states),('height',base.heights),('first_seen',base.first),('last_seen',base.last)]:np.testing.assert_array_equal(columns[name][ki],value)
                selected=np.flatnonzero(d['states']);ni=np.searchsorted(fc,codes(keys[selected]));np.testing.assert_array_equal(columns['state'][ni],d['states'][selected]);np.testing.assert_array_equal(columns['height'][ni],d['heights'][selected]);assert len(columns['cell'])==len(base.keys)+len(selected)and future['hypothetical']is True
                records.append(dict(start_node=sid,end_node=ex['end_node'],cm=cm.tolist(),baseline_cm=bc.tolist(),zero_new=not bool(ex['target']),predicted=len(selected),out_of_domain=len(missing),future_verified=True))
                if len(records)<=2:np.savez_compressed(dest/('%s_%02d.npz'%(rid,len(records))),keys=keys,y=y,prediction=d['states'],path=ex['trajectory'],gkeys=prepared.keys,gstates=prepared.values)
            assert content_key(graph)==before
        root=dict(root_id=rid,scene_id=row['scene_id'],records=records);save(dest/(rid+'.json'),root);roots.append(root);print('ROOT',len(roots),'windows',sum(len(r['records'])for r in roots),'seconds',time.perf_counter()-tick,flush=True)
    scenes=defaultdict(list);pooled=np.zeros((2,3,3),np.int64)
    for root in roots:
        cm=np.array([[e['baseline_cm'],e['cm']]for e in root['records']]);scenes[root['scene_id']].append(cm.mean(0));pooled+=cm.sum(0)
    mean=np.mean([np.mean(v,0)for v in scenes.values()],0);result=dict(role='LEGAL sensor-extension PUBLIC API on original fixed pilot; not full-development acceptance.',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),stride=stride,roots=len(roots),windows=sum(len(r['records'])for r in roots),hierarchical={k:extended_rates(c)for k,c in zip(['baseline','range'],mean)},pooled={k:extended_rates(c)for k,c in zip(['baseline','range'],pooled)},pooled_confusion=pooled.tolist(),seconds=time.perf_counter()-tick);save(dest/'summary.json',result);print('SUMMARY',result['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--stride',type=int,default=1);p.add_argument('--planar',action='store_true');p.add_argument('--proxy',action='store_true');p.add_argument('--spatial',action='store_true');p.add_argument('--radix',action='store_true');run(**vars(p.parse_args()))
