"""Original-domain pilot of LEGAL observed-map wall extrapolation."""
import argparse
import time
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .legal_camera import current_extrinsic,base_rotations
from .observed_walls import reconstruct,blocked_forecast
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


def run(extension=2.,roots=31):
    tick=time.perf_counter();m=read(HF/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']};rows,_=source_rows('dev');raw={r['root_id']:r for r in rows};records=[];details=[]
    for row in m['rows'][:roots]:
        legal=member(raw[row['root_id']],'legal_root.json.gz');current=legal['history'][-1];rootpath=np.array([[current['timestamp'],*current['robot_pose']]],np.float64);translation,rotation=current_extrinsic(rootpath,current['camera_origin'],current['camera_rotation']);a=load_arrays(row);old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
        with np.load(old['path'],allow_pickle=False)as z:x,y,prior=z['x'],z['y'],z['baseline']
        cache={};cms=[]
        for i,meta in enumerate(row['segments']):
            v=get_segment(a,i);path=v[3];res=float(v[-1][0]);gi=int(a['graph_index'][i]);e=lookup[meta['start_node'],meta['end_node']];sl=slice(e['start'],e['end']);keys=np.floor((x[sl,20:22].astype(np.float64)*10+path[0,[1,3]])/res).astype(np.int32);gt=y[sl]
            if gi not in cache:cache[gi]=reconstruct(v[0],v[1],v[2],res,extension=extension)
            segments=cache[gi];base=base_rotations(path[:,4]);origin=path[:,1:4]+base@translation;rot=base@rotation;lo=keys.min(0)-2;shape=keys.max(0)-lo+3;unknown=np.ones(tuple(shape),np.uint8);ij=v[0]-lo;valid=np.all((ij>=0)&(ij<shape),axis=1);at=ij[valid];unknown[at[:,0],at[:,1]]=0;state=blocked_forecast(segments,path,origin,rot,lo,shape,unknown,res);ij=keys-lo;geom=state[ij[:,0],ij[:,1]];p=prior[sl];free=(p[:,1]>p[:,2])&(p[:,1]+p[:,2]>=.2);hybrid=np.where(geom>0,2,np.where(free,1,0));pair=np.array([np.bincount(3*gt.astype(np.int64)+pred,minlength=9).reshape(3,3)for pred in [geom,hybrid]]);cms.append(pair);details.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],start_node=meta['start_node'],end_node=meta['end_node'],segments=len(segments),confusion=pair.tolist()))
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0)));print('ROOT',len(records),row['scene_id'],'seconds',time.perf_counter()-tick,flush=True)
    from collections import defaultdict
    scenes=defaultdict(list)
    for r in records:scenes[r['scene_id']].append(r['cm'])
    cm=np.mean([np.mean(values,axis=0)for values in scenes.values()],axis=0);pooled=np.sum([r['confusion']for r in details],axis=0);r=dict(role='DEPLOYABLE-LEGAL inputs: current map + current observed camera calibration + supplied trajectory only. Estimated optical walls from observed inflated BLOCKED; no hidden map, navmesh, future camera or actor truth.',extension=extension,roots=len(records),windows=len(details),hierarchical={k:extended_rates(c)for k,c in zip(['blocked_geometry_only','hybrid_baseline_free'],cm)},pooled={k:extended_rates(c)for k,c in zip(['blocked_geometry_only','hybrid_baseline_free'],pooled)},records=details,seconds=time.perf_counter()-tick);save(OUT/('observed_walls_extension_%g.json'%extension),r);print('SUMMARY',r['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extension',type=float,default=2.);p.add_argument('--roots',type=int,default=31);run(**vars(p.parse_args()))
