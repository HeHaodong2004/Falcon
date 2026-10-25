"""Exact real-crop differentials and paired contended K4/K8 pilot timings."""
import json
import time
import numpy as np
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.shared_crop_experiment import SharedCropPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.common import OUT,dump,sha,snapshot
from eventwm_graph.stage2_segment.features import SegmentFeatures
from eventwm_graph.stage2_segment.prepared import content_key


def equal(a,b):
    for field in ('candidate_cells','probabilities','states','heights','unknown_cells'):np.testing.assert_array_equal(a[field],b[field])
    assert a['static_updates']==b['static_updates']
    af=a['future_partial_graph'];bf=b['future_partial_graph']
    assert (af.to_snapshot() if hasattr(af,'to_snapshot') else af)==(bf.to_snapshot() if hasattr(bf,'to_snapshot') else bf)


def main(name='shared_crop_pilot',production=False):
    dest=OUT/name;dest.mkdir(exist_ok=False);checkpoint=OUT/'training/final/last.pt'
    models={label:cls(checkpoint,execution='optimized',future_format='native',device='cuda:0',shared_crops=label=='shared') for label,cls in [('reference',SegmentPredictor),('shared',SegmentPredictor if production else SharedCropPredictor)]}
    groups=inputs('multi');proof=[];records=[];holders={}
    dump(dest/'protocol.json',dict(source=snapshot(),checkpoint_sha256=sha(checkpoint),groups=len(groups),device='cuda:0',
        purpose='execution-only exact distance sharing, no model/threshold/domain change',accuracy='exact crop arrays and complete scientific outputs forK1/2/4/8',
        timing='fullAPI including shared prewarm and all requested futuregraphs; GCnormal; background8worker scientific evaluation onGPU1, NOT acceptance'))
    for index,group in enumerate(groups):
        graph=group['graph'];paths=[p['trajectory'] for p in group['paths']];original_key=content_key(graph)
        for k in (1,2,4,8):
            for model in models.values():model.clear_cache()
            outputs={name:model.predict_segments(partial_graph=graph,trajectories=paths[:k]) for name,model in models.items()}
            for a,b in zip(outputs['reference'],outputs['shared']):equal(a,b)
            a=models['reference'].prepare_graph(graph);b=models['shared'].prepare_graph(graph)
            for path in paths[:k]:
                ca=SegmentFeatures(graph,path,a);cb=SegmentFeatures(graph,path,b)
                np.testing.assert_array_equal(ca.lo,cb.lo);np.testing.assert_array_equal(ca.shape,cb.shape)
                np.testing.assert_array_equal(ca.states,cb.states);np.testing.assert_array_equal(ca.base_height.view(np.uint32),cb.base_height.view(np.uint32))
                for x,y in zip(ca.dist,cb.dist):np.testing.assert_array_equal(x.view(np.uint32),y.view(np.uint32))
            proof.append(dict(root_id=group['root_id'],k=k,exact=True,batch_profile=dict(models['shared'].last_batch_profile)))
        assert original_key==content_key(graph)
        for k in (4,8):
            for condition in ('map_cold','retained_cache'):
                for repeat in range(4):
                    for name in (('reference','shared') if (index+repeat)%2 else ('shared','reference')):
                        model=models[name]
                        if condition=='map_cold':model.clear_cache()
                        else:model.predict_segments(partial_graph=graph,trajectories=paths[:k])
                        cpu=time.process_time();tick=time.perf_counter();holders[name]=model.predict_segments(partial_graph=graph,trajectories=paths[:k]);wall=time.perf_counter()-tick
                        records.append(dict(model=name,k=k,condition=condition,root_id=group['root_id'],seconds=wall,cpu_seconds=time.process_time()-cpu,
                            profile=dict(model.last_batch_profile),materialized_future_graphs=len(holders[name])))
        if index%8==0:print(json.dumps(dict(groups=index+1,total=len(groups),proof=len(proof))),flush=True)
    summary={}
    for name in models:
        summary[name]={}
        for k in (4,8):
            summary[name][str(k)]={}
            for condition in ('map_cold','retained_cache'):
                rows=[r for r in records if r['model']==name and r['k']==k and r['condition']==condition]
                summary[name][str(k)][condition]=dict(p50_ms=float(np.median([r['seconds'] for r in rows]))*1000,
                    p95_ms=float(np.quantile([r['seconds'] for r in rows],.95))*1000,
                    mean_cpu_utilization_percent=100*sum(r['cpu_seconds'] for r in rows)/sum(r['seconds'] for r in rows))
    dump(dest/'results.json',dict(summary=summary,proof=proof,calls=records,scientific_arrays_and_graphs_exact=True,
        crop_states_height_and_distances_bitwise_exact=True,not_acceptance=True,background='full scientific evaluation8workers/GPU1',source=snapshot()))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='shared_crop_pilot');parser.add_argument('--production',action='store_true');main(**vars(parser.parse_args()))
