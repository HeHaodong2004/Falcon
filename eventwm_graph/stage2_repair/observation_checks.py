"""Recording-backed certificate/frustum and collector version diagnosis only."""
from collections import Counter
import math
from pathlib import Path
import numpy as np
from .diagnose import OUT,HF,read,save,sha,source_rows,member


def frustum(points,origin,rotation):
    local=(points-origin)@rotation;forward=-local[...,2]
    return (forward>1e-6)&(abs(local[...,0])<=forward)&(abs(local[...,1])<=forward)&(np.linalg.norm(local,axis=-1)<=10.)


def main():
    train,_=source_rows('train');versions=Counter(r.get('version_set','missing')for r in train);representatives={}
    for row in train:representatives.setdefault(row.get('version_set','missing'),row)
    source_rows_info=[]
    for version,row in representatives.items():
        legal=member(row,'legal_root.json.gz');source_rows_info.append(dict(version_set=version,roots=versions[version],
            observer_sources={k:v for k,v in legal['version'].items()if 'observer' in k or k=='acceleration/acceleration.py'},sensor=legal['sensor']))
    rows,_=source_rows('dev');index={r['root_id']:r for r in rows};pilot=read(HF/'light_eval_cache/manifest.json');cal=read(OUT/'geometry_cohorts.json')['training_calibration']['mean'];counts=Counter();recencies=[]
    for rr in pilot['rows']:
        row=index[rr['root_id']];legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');r=legal['partial_graph']['resolution']
        # Exact 5 square probes +16 clearance halo points, at3heights.
        square=np.array([[0,0,0],[-.49*r,0,-.49*r],[-.49*r,0,.49*r],[.49*r,0,-.49*r],[.49*r,0,.49*r]])
        halo=np.array([[math.cos(a)*(.25+r/math.sqrt(2)),0,math.sin(a)*(.25+r/math.sqrt(2))]for a in np.linspace(0,2*math.pi,16,endpoint=False)])
        planar=np.concatenate([square,halo]);offsets=np.concatenate([planar+[0,h,0]for h in [.025,.7,1.4]])
        for ob in legal['history']+[n['observation']for n in nodes]:
            free=[u for u in ob['static_updates']if u['state']==1 and u['source']!='occupied_ego_base_support']
            if not free:continue
            centers=np.array([[(u['cell'][0]+.5)*r,u['height'],(u['cell'][1]+.5)*r]for u in free]);targets=centers[:,None,:]+offsets
            actual=frustum(targets,np.array(ob['camera_origin']),np.array(ob['camera_rotation'])).all(1)
            base=np.array(ob['robot_pose']);yaw=base[3];c=math.cos(yaw);s=math.sin(yaw)
            # Raw base-center horizontal feature proxy (heading is supplied).
            delta=centers[:,[0,2]]-base[[0,2]];distance=np.linalg.norm(delta,axis=1);facing=(delta[:,0]*c-delta[:,1]*s)/np.maximum(distance,1e-9)
            proxy=(distance<=10)&(facing>=math.cos(math.pi/4))
            # Fixed extrinsic estimate from TRAIN history only; diagnostic,
            # never substituted into production inputs in this run.
            origin=base[:3]+[cal[0]*c+cal[2]*s,cal[1],-cal[0]*s+cal[2]*c]
            ya=yaw+cal[3];pitch=cal[4];forward=np.array([math.cos(pitch)*math.cos(ya),math.sin(pitch),-math.cos(pitch)*math.sin(ya)])
            right=np.array([math.sin(ya),0,math.cos(ya)]);up=np.cross(right,forward);rotation=np.column_stack([right,up,-forward])
            calibrated=frustum(targets,origin,rotation).all(1)
            counts.update(published_free=len(free),actual_all63_pass=int(actual.sum()),base_horizontal_proxy_pass=int(proxy.sum()),fixed_calibration_all63_pass=int(calibrated.sum()))
        graph=legal['partial_graph'];freeages=[graph['timestamp']-c['last_seen']for c in graph['cells']if c['state']==1];recencies.extend(freeages)
    # Verify the source identity rather than assuming today's observer collected it.
    archive=Path('/home/haodong/workspace/projects/eventwm-falcon-acceleration-v2/output/stage01_acceleration_v2_20261021/candidate_source')
    hashes={v for row in source_rows_info for k,v in row['observer_sources'].items()if k.endswith('fast_observer.py')};resolved=[]
    for h in hashes:
        for p in archive.glob('*/eventwm_graph/partial/fast_observer.py'):
            if sha(p)==h:resolved.append(dict(sha256=h,path=str(p)));break
    save(OUT/'observation_checks.json',dict(training_versions=source_rows_info,resolved_archived_observers=resolved,pilot_free_probe_check=dict(counts),
        free_certificate_age_quantiles_s=np.quantile(recencies,[.05,.5,.95]).tolist(),
        scope='Frustum disagreement on actually published FREE certificates, not counterfactual visibility/occlusion frequency. Fixed calibration from selected TRAIN history only; production inputs unchanged.'))
    print('certificate check',dict(counts),'collector versions',len(source_rows_info),flush=True)


if __name__=='__main__':main()
