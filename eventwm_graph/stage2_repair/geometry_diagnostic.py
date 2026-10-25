"""Offline-only camera/cohort diagnostics. Privileged metadata is never an input."""
from collections import Counter, defaultdict
import math
import numpy as np
from .diagnose import OUT, HF, read, save, member, source_rows
from eventwm_graph.stage2_hf.spatial_eval import score
from scripts.stage2_diagnostic.common import extended_rates


def camera(ob):
    base=np.asarray(ob['robot_pose']);origin=np.asarray(ob['camera_origin']);r=np.asarray(ob['camera_rotation']);f=-r[:,2]
    c=math.cos(base[3]);s=math.sin(base[3]);d=origin-base[:3]
    return np.array([d[0]*c-d[2]*s,d[1],d[0]*s+d[2]*c,
        math.atan2(-f[2],f[0])-base[3],math.asin(np.clip(f[1],-1,1)),math.atan2(r[1,0],r[1,1])])


def describe(a):
    a=np.asarray(a)
    return dict(n=len(a),mean=a.mean(0).tolist(),std=a.std(0).tolist(),
        p05=np.quantile(a,.05,axis=0).tolist(),p50=np.quantile(a,.5,axis=0).tolist(),p95=np.quantile(a,.95,axis=0).tolist(),min=a.min(0).tolist(),max=a.max(0).tolist())


def main():
    dev,_=source_rows('dev');index={r['root_id']:r for r in dev}
    manifest=read(HF/'light_eval_cache/manifest.json');records=[];allcamera=[];sources=Counter();protocols=Counter();calibration=[]
    # A small fixed training cohort estimates extrinsics; development is diagnostic only.
    train,_=source_rows('train');trainindex={r['root_id']:r for r in train}
    trace=read(OUT/'trace.json')
    for rid in sorted({r['root_id'] for r in trace['examples']}):
        row=trainindex[rid];legal=member(row,'legal_root.json.gz')
        for ob in legal['history']:
            v=camera(ob);v[3]=math.atan2(math.sin(v[3]),math.cos(v[3]));calibration.append(v)
    for item in manifest['rows']:
        row=index[item['root_id']];legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz')
        # Explicit episode human_num, not tracks. Read offline initialization solely here.
        initialization=member(row,'privileged_initialization.json.gz')
        human_count=initialization['episode']['info']['human_num']
        for k,v in legal['version'].items():
            if 'observer' in k or k=='acceleration/acceleration.py':sources[(k,v)]+=1
        cam=[]
        for ob in legal['history']+[n['observation'] for n in nodes]:
            protocols[ob.get('sensor_protocol','missing')]+=1
            v=camera(ob);v[3]=math.atan2(math.sin(v[3]),math.cos(v[3]));cam.append(v);allcamera.append(v)
        with np.load(item['path'],allow_pickle=False)as z:p=z['baseline'];y=z['y']
        for meta in item['segments']:
            sl=slice(meta['start'],meta['end']);cm,_,_=score(p[sl],y[sl]);sid=meta['start_node'];eid=meta['end_node'];anc=[]
            while eid!=sid:
                ob=nodes[eid]['observation'];anc.append(ob);eid=nodes[eid]['parent']
            anc.reverse();first=legal['history'][-1] if sid is None else nodes[sid]['observation'];path=np.array([first['robot_pose']]+[ob['robot_pose'] for ob in anc]);mov=float(np.linalg.norm(np.diff(path[:,[0,2]],axis=0),axis=1).sum());dy=np.diff(path[:,3]);rot=float(np.abs(np.arctan2(np.sin(dy),np.cos(dy))).sum())
            movement='mixed' if mov>1e-4 and rot>1e-4 else 'translation' if mov>1e-4 else 'rotation' if rot>1e-4 else 'stationary'
            cr=np.array([camera(ob) for ob in anc]) if anc else np.empty((0,6));yawerr=np.abs(np.arctan2(np.sin(cr[:,3]),np.cos(cr[:,3])))
            records.append(dict(root_id=item['root_id'],scene_id=item['scene_id'],start_node=sid,end_node=meta['end_node'],human_count=human_count,
                duration=meta['duration'],new_cells=meta['new_cells'],movement=movement,cm=cm[6].tolist(),
                yaw_error_deg=float(np.degrees(yawerr).mean()) if len(yawerr) else 0.,
                mean_offset_m=float(np.linalg.norm(cr[:,:3],axis=1).mean()) if len(cr) else 0.))
    groups={}
    for name,rs in [('zero_human',[r for r in records if r['human_count']==0]),('human_present',[r for r in records if r['human_count']>0])]:
        if not rs:groups[name]=dict(windows=0);continue
        cm=np.sum([r['cm'] for r in rs],axis=0);groups[name]=dict(windows=len(rs),roots=len({r['root_id']for r in rs}),scenes=len({r['scene_id']for r in rs}),
            pooled=extended_rates(cm),cm=cm.tolist(),duration_mean=float(np.mean([r['duration']for r in rs])),target_mean=float(np.mean([r['new_cells']for r in rs])),motions=dict(Counter(r['movement']for r in rs)))
    # Direct standardized descriptive comparison within common scene/motion/duration/size strata.
    bins=defaultdict(lambda:defaultdict(list))
    for r in records:
        key=(r['scene_id'],r['movement'],int(np.digitize(r['duration'],[.1,.5,1,2])),int(np.digitize(r['new_cells'],[1,100,500])))
        bins[key][r['human_count']==0].append(r)
    matched=[]
    for key,b in bins.items():
        if True in b and False in b:
            matched.append(dict(stratum=list(key),zero_windows=len(b[True]),human_windows=len(b[False]),
                zero_cm=np.mean([r['cm']for r in b[True]],axis=0).tolist(),human_cm=np.mean([r['cm']for r in b[False]],axis=0).tolist()))
    matched_result=dict(common_strata=len(matched),strata=matched)
    if matched:
        for name,key in [('zero', 'zero_cm'),('human','human_cm')]:matched_result[name]=extended_rates(np.mean([r[key]for r in matched],axis=0))
    save(OUT/'geometry_cohorts.json',dict(scope='Pilot descriptive, same frozen checkpoint at threshold .20. Human metadata and future camera used only offline. No causal occlusion claim.',
        sources=[dict(file=k[0],sha256=k[1],roots=v) for k,v in sorted(sources.items())],protocols=dict(protocols),
        camera_columns=['forward_offset_m','height_offset_m','lateral_offset_m','camera_minus_base_yaw_rad','pitch_rad','roll_rad'],
        training_calibration=describe(calibration),pilot_camera=describe(allcamera),groups=groups,matched=matched_result,windows=records))
    print('camera',describe(allcamera),flush=True);print('cohorts',groups,'matched',matched_result,flush=True)


if __name__=='__main__':main()
