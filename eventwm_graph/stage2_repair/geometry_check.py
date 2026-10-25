"""Native-recording/GPU comparison, with full independent scene preparation."""
import argparse
import copy
import time
import numpy as np
from .geometry_api import GeometryPredictor
from .geometry_reference import trial_cameras
from .diagnose import OUT, read, save, source_rows, member
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.update import apply_graph_delta
from scripts.stage2_diagnostic.common import extended_rates


def run(device='cuda:0', windows=8, name='geometry_gpu_probe'):
    source=read(OUT/'geometry/native_geometry_probe/report.json')['windows'][:windows]
    rows,_=source_rows('dev');index={r['root_id']:r for r in rows};wanted={}
    for i,e in enumerate(source,1):wanted.setdefault(e['root_id'],{})[e['start_node'],e['end_node']]=i
    reports=[]
    for rid, targets in wanted.items():
        row=index[rid];legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');ep=member(row,'privileged_initialization.json.gz')['episode'];tick=time.perf_counter()
        m=GeometryPredictor(ep['scene_id'],ep['info']['graph_navmesh'],legal['partial_graph']['resolution'],legal['sensor'],device=device)
        print('SCENE',row['scene_id'],'certificates',len(m.asset.arrays['keys']),'triangles',len(m.asset.arrays['triangles']),'startup',time.perf_counter()-tick,flush=True)
        try:
            for ex in replay(row,'mixed'):
                if 'audit'in ex or (ex['start_node'],ex['end_node'])not in targets:continue
                i=targets[ex['start_node'],ex['end_node']];p,o,r=trial_cameras(legal,nodes,ex['start_node'],ex['end_node']);before=copy.deepcopy(ex['graph']);tick=time.perf_counter();delta=m.predict(partial_graph=ex['graph'],trajectory=p,camera_origins=o,camera_rotations=r);seconds=time.perf_counter()-tick
                with np.load(OUT/'geometry/native_geometry_probe'/('%03d.npz'%i),allow_pickle=False)as z:expected=z['prediction'];np.testing.assert_array_equal(delta['candidate_cells'],z['keys'])
                y,_,missing=labels(delta['candidate_cells'],ex['target']);cm=np.bincount(3*y.astype(np.int64)+delta['states'],minlength=9).reshape(3,3)
                for s,h in missing:cm[s,0]+=1
                assert ex['graph']==before
                assert delta['future_partial_graph'].to_snapshot()==apply_graph_delta(ex['graph'],delta)
                report=dict(root_id=rid,scene_id=row['scene_id'],start_node=ex['start_node'],end_node=ex['end_node'],gpu_native_different_cells=int(np.count_nonzero(delta['states']!=expected)),metrics=extended_rates(cm),confusion=cm.tolist(),complete_seconds=seconds,profile=m.last_batch_profile)
                reports.append(report);print('RESULT',report,flush=True);save(OUT/'geometry'/name/'report.json',dict(input_contract='REVISED static scene geometry + trial camera poses. GPU intersection approximation measured against native reference and original certificates.',windows=reports))
        finally:m.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--windows',type=int,default=8);p.add_argument('--name',default='geometry_gpu_probe');run(**vars(p.parse_args()))
