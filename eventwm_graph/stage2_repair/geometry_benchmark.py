"""OFFLINE ORACLE latency only; not real legal-input WM acceptance."""
import argparse
import gc
import os
import subprocess
import time
import numpy as np
import torch
from .diagnose import OUT, CKPT, save, source_rows, member
from .geometry_api import GeometryPredictor
from .increment import SparseIncrement
from .benchmark import summary
from eventwm_graph.stage2_hf.baseline import ThresholdBaseline
from eventwm_graph.stage2_segment.benchmark import inputs


def run(device='cuda:1',repetitions=3,groups=32,name='geometry_complete'):
    torch.set_num_threads(1);gc.enable();work=inputs('multi');rows,_=source_rows('dev');index={r['root_id']:r for r in rows}
    order=sorted(range(len(work)),key=lambda i:len(work[i]['graph']['cells']));dense=set(order[len(order)//2:])
    if groups<len(order):order=[order[int(i)]for i in np.linspace(0,len(order)-1,groups)]
    baseline=ThresholdBaseline(CKPT,threshold=.2,execution='optimized',future_format='native',device=device,feature_threads=1)
    calls=[];startup=[];load=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,pmem,args','--sort=-pcpu'],text=True).splitlines()[:25]
    for wi in order:
        item=work[wi];row=index[item['root_id']];legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');ep=member(row,'privileged_initialization.json.gz')['episode'];tick=time.perf_counter()
        model=GeometryPredictor(ep['scene_id'],ep['info']['graph_navmesh'],item['graph']['resolution'],legal['sensor'],device=device)
        scene_startup=time.perf_counter()-tick;paths=[np.array(p['trajectory'],np.float64)for p in item['paths'][:6]];camera=[]
        for path,p in zip(paths,item['paths'][:6]):
            obs=[legal['history'][-1]]+[nodes[i]['observation']for i in p['node_ids']]
            assert len(obs)==len(path)
            camera.append(dict(origins=np.array([o['camera_origin']for o in obs]),rotations=np.array([o['camera_rotation']for o in obs])))
        try:
            for label,m in [('baseline',baseline),('candidate',model)]:
                tick=time.perf_counter()
                if label=='candidate':m.predict_segments(partial_graph=item['graph'],trajectories=paths,camera_trajectories=camera)
                else:m.predict_segments(partial_graph=item['graph'],trajectories=paths)
                torch.cuda.synchronize(device);startup.append(dict(group=wi,model=label,scene_prepare_seconds=scene_startup if label=='candidate'else 0.,first_K6_seconds=time.perf_counter()-tick))
            for rep in range(repetitions):
                for k in [1,4,6]:
                    scientific={}
                    for label,m in [('baseline',baseline),('candidate',model)]:
                        for cache in ['cold','warm']:
                            if cache=='cold':m.clear_cache()
                            else:
                                if label=='candidate':m.predict_segments(partial_graph=item['graph'],trajectories=paths[:k],camera_trajectories=camera[:k])
                                else:m.predict_segments(partial_graph=item['graph'],trajectories=paths[:k])
                            torch.cuda.synchronize(device);tick=time.perf_counter()
                            if label=='candidate':result=m.predict_segments(partial_graph=item['graph'],trajectories=paths[:k],camera_trajectories=camera[:k])
                            else:
                                result=m.predict_segments(partial_graph=item['graph'],trajectories=paths[:k])
                                for delta in result:
                                    chosen=np.flatnonzero(delta['states']);delta['static_updates']=SparseIncrement(delta['candidate_cells'][chosen],delta['states'][chosen],delta['heights'][chosen],delta['timestamp'])
                            torch.cuda.synchronize(device);ms=(time.perf_counter()-tick)*1000
                            assert len(result)==k and all('future_partial_graph'in d for d in result)
                            scientific[label]=[d['candidate_cells']for d in result]
                            calls.append(dict(model=label,cohort='dense'if wi in dense else'representative',cache=cache,k=k,group=wi,repetition=rep,ms=ms,candidates=[len(d['candidate_cells'])for d in result],profile=dict(m.last_batch_profile)))
                    for a,b in zip(scientific['baseline'],scientific['candidate']):np.testing.assert_array_equal(a,b)
            save(OUT/('benchmark_'+name+'.json'),dict(oracle_only=True,acceptance_eligible=False,contract='ORACLE ONLY privileged geometry/camera engine versus original base-trajectory model. These inputs are unavailable to the real WM. Identical original candidate universe; not a deployable latency/accuracy result.',
                scope='Timer includes graph/input preparation, candidates, camera packing, host/device transfers/synchronization, geometry/probability/hard decoding, concrete sparse increments, normal complete hypothetical graphs. Serialization EXCLUDED. No prediction/view output caching. Cold=current graph cache empty; warm=current graph reusable; scene assets resident in both cases.',
                groups=len(order),complete_groups=len({r['group']for r in calls}),repetitions=repetitions,device=device,hardware=torch.cuda.get_device_name(device),precision='Geometry F64 frusta/certificates and F32 BVH intersections; baseline F32 neural/features, TF32 disabled.',concurrent_load=load,loadavg=os.getloadavg(),startup=startup,summary=summary(calls),calls=calls))
            print('GROUP',wi,'finished','K6',[(r['model'],r['cache'],round(r['ms'],2))for r in calls if r['group']==wi and r['k']==6],flush=True)
        finally:model.close()
    print('SUMMARY',summary(calls),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--repetitions',type=int,default=3);p.add_argument('--groups',type=int,default=32);p.add_argument('--name',default='geometry_complete');run(**vars(p.parse_args()))
