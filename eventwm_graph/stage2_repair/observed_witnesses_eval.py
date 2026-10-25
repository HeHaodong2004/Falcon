"""Original-domain pilot using current/past LEGAL certificate witnesses."""
import argparse
from collections import defaultdict
import time
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .legal_camera import current_extrinsic,base_rotations
from .observed_walls import blocked_forecast
from .observed_witnesses import decode,segments
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


def run(extension=2.,roots=31):
    tick=time.perf_counter();m=read(HF/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']};rows,_=source_rows('dev');raw={r['root_id']:r for r in rows};records=[];details=[];stats=np.zeros(3,np.int64)
    for row in m['rows'][:roots]:
        source=raw[row['root_id']];legal=member(source,'legal_root.json.gz');nodes=member(source,'future_observations.json.gz');current=legal['history'][-1];rootpath=np.array([[current['timestamp'],*current['robot_pose']]],np.float64);translation,rotation=current_extrinsic(rootpath,current['camera_origin'],current['camera_rotation']);a=load_arrays(row);res=float(a['geometry'][0]);history=[]
        for ob in legal['history']:
            pp,st=decode(ob,res);history.append(pp);stats+=[st['rows'],st['recovered_rows'],st['patches']]
        rootpoints=np.concatenate(history);point_cache={None:rootpoints};old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
        with np.load(old['path'],allow_pickle=False)as z:x,y,prior=z['x'],z['y'],z['baseline']
        def past_points(node):
            if node not in point_cache:
                parent=past_points(nodes[node]['parent']);pp,st=decode(nodes[node]['observation'],res);stats[:]+=[st['rows'],st['recovered_rows'],st['patches']];point_cache[node]=np.concatenate([parent,pp])
            return point_cache[node]
        cache={};cms=[]
        for i,meta in enumerate(row['segments']):
            v=get_segment(a,i);path=v[3];gi=int(a['graph_index'][i]);e=lookup[meta['start_node'],meta['end_node']];sl=slice(e['start'],e['end']);keys=np.floor((x[sl,20:22].astype(np.float64)*10+path[0,[1,3]])/res).astype(np.int32);gt=y[sl]
            if gi not in cache:cache[gi]=segments(past_points(meta['start_node']),v[0][v[1]==1],res,extension=extension)
            walls=cache[gi];base=base_rotations(path[:,4]);origin=path[:,1:4]+base@translation;rot=base@rotation;lo=keys.min(0)-2;shape=keys.max(0)-lo+3;unknown=np.ones(tuple(shape),np.uint8);ij=v[0]-lo;valid=np.all((ij>=0)&(ij<shape),axis=1);at=ij[valid];unknown[at[:,0],at[:,1]]=0;state=blocked_forecast(walls,path,origin,rot,lo,shape,unknown,res);ij=keys-lo;geom=state[ij[:,0],ij[:,1]];p=prior[sl];free=(p[:,1]>p[:,2])&(p[:,1]+p[:,2]>=.2);hybrid=np.where(geom>0,2,np.where(free,1,0));pair=np.array([np.bincount(3*gt.astype(np.int64)+pred,minlength=9).reshape(3,3)for pred in [geom,hybrid]]);cms.append(pair);details.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],start_node=meta['start_node'],end_node=meta['end_node'],segments=len(walls),confusion=pair.tolist()))
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0)));print('ROOT',len(records),row['scene_id'],'recovered',stats.tolist(),'seconds',time.perf_counter()-tick,flush=True)
    scenes=defaultdict(list)
    for r in records:scenes[r['scene_id']].append(r['cm'])
    cm=np.mean([np.mean(values,axis=0)for values in scenes.values()],axis=0);pooled=np.sum([r['confusion']for r in details],axis=0);result=dict(role='LEGAL current/past observation footprints + current camera calibration + supplied trajectory only; no hidden scene/navmesh or future camera inputs. Past DAG ancestry only, never sibling/future observations.',extension=extension,roots=len(records),windows=len(details),footprint_stats=dict(blocked_rows=int(stats[0]),recovered_rows=int(stats[1]),witnesses=int(stats[2])),hierarchical={k:extended_rates(c)for k,c in zip(['blocked_witness_geometry','hybrid_baseline_free'],cm)},pooled={k:extended_rates(c)for k,c in zip(['blocked_witness_geometry','hybrid_baseline_free'],pooled)},records=details,seconds=time.perf_counter()-tick);save(OUT/('observed_witnesses_extension_%g.json'%extension),result);print('SUMMARY',result['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extension',type=float,default=2.);p.add_argument('--roots',type=int,default=31);run(**vars(p.parse_args()))
