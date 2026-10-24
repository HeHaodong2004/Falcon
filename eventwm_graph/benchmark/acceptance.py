"""Grade actual-action evidence against benchmark navigation, not bit equality.

The first certification used 0.2 mm as both a model parity diagnostic and an
execution verdict. Long float32 native/Bullet rollouts can differ by >0.2 mm
while actually reaching the target on the correct connected surface. This
grader checks the saved *actual* trajectory independently and separately bounds
compiler prediction error at 1 mm. Neither bound changes collision labels.
"""
import copy
import os
from pathlib import Path
import time
import numpy as np
from ..config import digest,read_json,write_json,file_digest
from ..map import load_hierarchical_map
from .certify import VERIFIED,UNKNOWN,graph_pairs,independent_trace_geometry

VERSION='actual-surface-goal-evidence-1'


def grade(source_manifest,output):
    started=time.perf_counter();source_manifest=Path(source_manifest).resolve()
    m=read_json(source_manifest);folder=source_manifest.parent
    for filename,sha in m['checksums'].items():
        if file_digest(folder/filename)!=sha:raise ValueError('Damaged source certification evidence')
    h=load_hierarchical_map(m['source_manifest'])
    identity=copy.deepcopy(m['identity'])
    identity['acceptance']={'version':VERSION,'implementation':file_digest(__file__),
        'source_certificate':m['map_id'],'source_manifest_sha256':file_digest(source_manifest),
        'compiler_position_tolerance_m':.001,'compiler_yaw_tolerance_rad':.0002,
        'actual_goal_tolerance_m':.16,'actual_heading_tolerance_rad':10/120/2+.0002}
    map_id=digest(identity);dest=Path(output)/m['scene_id']/map_id;dest.mkdir(parents=True,exist_ok=True)
    (dest/'traces').mkdir(exist_ok=True)
    edges=read_json(folder/'edges.json');routes=read_json(folder/'routes.json')
    for row in [*edges,*routes]:
        row['source_certificate_status']=row['status'];row['map_id']=map_id
        if 'trace' not in row:continue  # A failed search remains UNKNOWN.
        source=folder/'traces'/row['trace'];target=dest/'traces'/row['trace']
        if not target.exists():os.link(source,target)
        if file_digest(target)!=row['trace_sha256']:raise ValueError('Trace identity mismatch')
        with np.load(target,allow_pickle=False) as a:
            actual=a['actual_poses'];predicted=a['predicted_poses']
        endpoint=np.asarray(row['path'][-1] if 'path' in row else h.nodes[row['target_node']]['position'])
        goal_error=float(np.linalg.norm(actual[-1,:3]-endpoint))
        yaw_error=float(abs(np.arctan2(np.sin(actual[-1,3]),np.cos(actual[-1,3]))))
        native=independent_trace_geometry(h,actual)
        prediction_error=float(np.linalg.norm(actual[:,:3]-predicted[:,:3],axis=1).max())
        prediction_yaw=float(np.max(abs(np.arctan2(np.sin(actual[:,3]-predicted[:,3]),np.cos(actual[:,3]-predicted[:,3])))))
        valid=(native['valid'] and goal_error<=.16 and yaw_error<=10/120/2+.0002 and
               prediction_error<=.001 and prediction_yaw<=.0002 and not row['actual']['benchmark_should_end'])
        row['status']=VERIFIED if valid else UNKNOWN
        row['reason']='actual_surface_and_goal_verified' if valid else 'actual_acceptance_inconclusive'
        row['actual_acceptance']={'valid':bool(valid),'goal_error_m':goal_error,'heading_error_rad':yaw_error,
            'prediction_error_m':prediction_error,'prediction_yaw_error_rad':prediction_yaw,
            'surface':native,'compiler_position_tolerance_m':.001,'version':VERSION}
    original=[{'source':e['source'],'target':e['target']} for e in h.edges.values()]
    original+=[{'source':e['target'],'target':e['source']} for e in h.edges.values()]
    _,before=graph_pairs(h.nodes,original);_,after=graph_pairs(h.nodes,[e for e in edges if e['status']==VERIFIED])
    report=copy.deepcopy(m['report']);report['map_id']=map_id
    report.update(status_counts={status:sum(e['status']==status for e in edges) for status in (VERIFIED,UNKNOWN)},
        certified_reachable_pairs=int(after.sum()),lost_reachable_pairs=int((before&~after).sum()),
        multi_edge_passed=sum(r['status']==VERIFIED for r in routes),
        known_false_positive_executable_edges=sum(e['status']==VERIFIED and not e.get('actual_acceptance',{}).get('valid',False) for e in edges),
        original_certification=m['map_id'],acceptance_grading_seconds=time.perf_counter()-started,
        maximum_actual_prediction_error_m=max(e.get('actual_acceptance',{}).get('prediction_error_m',0) for e in [*edges,*routes]),
        numerical_failures_reclassified=sum(e['source_certificate_status']!=VERIFIED and e['status']==VERIFIED for e in [*edges,*routes]))
    report['production_gate']=(report['known_false_positive_executable_edges']==0 and
        report['lost_reachable_pairs']==0 and report['multi_edge_passed']==len(routes) and bool(edges))
    report['gate_scope']='Actual native surface, achieved position/orientation, complete directed edge evidence, route composition; compiler error <=1 mm, not a body-clearance margin'
    write_json(dest/'edges.json',edges);write_json(dest/'routes.json',routes);write_json(dest/'report.json',report);write_json(dest/'identity.json',identity)
    record={**m,'identity':identity,'map_id':map_id,'report':report,
        'acceptance_version':VERSION,'checksums':{str(p.relative_to(dest)):file_digest(p) for p in sorted((dest/'traces').glob('*.npz'))}}
    record['checksums'].update({n:file_digest(dest/n) for n in ('edges.json','routes.json','report.json','identity.json')})
    write_json(dest/'manifest.json',record);write_json(dest.parent/'latest.json',{'manifest':str((dest/'manifest.json').resolve()),'map_id':map_id})
    return record
