"""OFFLINE oracle geometry with only a LEGAL current-camera proxy.

This does not make geometry available to a deployed WM. It isolates camera
uncertainty before training a legal-input student; predictions receive no
recorded future camera poses. Supervision remains the original certificates.
"""
import argparse
from collections import defaultdict
import time
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .geometry_api import GeometryPredictor
from .legal_camera import proxy_cameras
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay
from scripts.stage2_diagnostic.common import extended_rates


def run(device='cuda:0',roots=31):
    rows,_=source_rows('dev');selected=set(read(HF/'protocol.json')['pilot_selection_roots']);rows=[r for r in rows if r['root_id']in selected][:roots];tick=time.perf_counter();records=[];scene_rows=defaultdict(list)
    for row in rows:scene_rows[row['scene_id']].append(row)
    for scene,rr in sorted(scene_rows.items()):
        engine=None
        try:
            for row in rr:
                legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');ep=member(row,'privileged_initialization.json.gz')['episode']
                if engine is None:engine=GeometryPredictor(ep['scene_id'],ep['info']['graph_navmesh'],legal['partial_graph']['resolution'],legal['sensor'],device=device)
                for ex in replay(row,'mixed'):
                    if 'audit'in ex:continue
                    current=legal['history'][-1]if ex['start_node']is None else nodes[ex['start_node']]['observation'];p=ex['trajectory'];o,r=proxy_cameras(p,current['camera_origin'],current['camera_rotation']);delta=engine.predict(partial_graph=ex['graph'],trajectory=p,camera_origins=o,camera_rotations=r);y,_,missing=labels(delta['candidate_cells'],ex['target']);cm=np.bincount(3*y.astype(np.int64)+delta['states'],minlength=9).reshape(3,3)
                    for state,height in missing:cm[state,0]+=1
                    records.append(dict(root_id=row['root_id'],scene_id=scene,start_node=ex['start_node'],end_node=ex['end_node'],confusion=cm.tolist()))
                print(scene,len(records),'windows',time.perf_counter()-tick,flush=True)
        finally:
            if engine is not None:engine.close()
    roots_cm=defaultdict(list);root_scene={r['root_id']:r['scene_id']for r in records}
    for r in records:roots_cm[r['root_id']].append(r['confusion'])
    scenes=defaultdict(list)
    for rid,cm in roots_cm.items():scenes[root_scene[rid]].append(np.mean(cm,axis=0))
    cm=np.mean([np.mean(values,axis=0)for values in scenes.values()],axis=0);result=dict(role='OFFLINE DIAGNOSTIC ONLY: privileged geometry, legal camera0 rigid-extrinsic proxy; no future camera pose input. Not a deployment score.',windows=len(records),roots=len(roots_cm),scenes=len(scenes),hierarchical=extended_rates(cm),pooled=extended_rates(np.sum([r['confusion']for r in records],axis=0)),records=records,seconds=time.perf_counter()-tick);save(OUT/'geometry/legal_current_camera_oracle.json',result);print('SUMMARY',result['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--roots',type=int,default=31);run(**vars(p.parse_args()))
