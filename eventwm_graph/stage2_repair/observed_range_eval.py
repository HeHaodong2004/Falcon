"""Bounded legal range-input reconstruction test on ALL windows of its Roots."""
import argparse
from collections import defaultdict
import time
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .observed_range_mesh import ObservedRangeMesh
from .legal_camera import current_extrinsic,base_rotations
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


def run(device='cuda:0',stride=1,tracked_filter=False,free_projection=False,plane_extension=0.,branch_current=False,floor_support=False,gpu_mesh=False,roots=31,cspace_filter=False):
    tick=time.perf_counter();m=read(HF/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']};rows,_=source_rows('dev');raw={r['root_id']:r for r in rows};records=[];details=[];prep=[]
    mesh_type=ObservedRangeMesh
    if gpu_mesh:
        from .range_gpu import GpuObservedRangeMesh
        mesh_type=GpuObservedRangeMesh
    for row in m['rows'][:roots]:
        rp=OUT/('range_branches'if branch_current else'range_replay')/(row['root_id']+'.npz')
        if not rp.exists():continue
        legal=member(raw[row['root_id']],'legal_root.json.gz');current=legal['history'][-1];rootpath=np.array([[current['timestamp'],*current['robot_pose']]],np.float64);translation,rotation=current_extrinsic(rootpath,current['camera_origin'],current['camera_rotation']);a=load_arrays(row);res=float(a['geometry'][0]);old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
        nodes=member(raw[row['root_id']],'future_observations.json.gz')if branch_current else None
        tracks=[o['tracks']for o in legal['history']]if tracked_filter else None
        if branch_current and tracked_filter:tracks += [n['observation']['tracks']for n in nodes]
        free_keys=np.array([c['cell']for c in legal['partial_graph']['cells']if c['state']==1],np.int32).reshape(-1,2)
        with np.load(rp,allow_pickle=False)as z:sensors={k:z[k]for k in z.files}
        if not branch_current:
            mesh=mesh_type(sensors['depth'],sensors['origins'],sensors['rotations'],sensors['timestamps'],current['timestamp'],res,device,stride,tracks,plane_extension,free_keys)
            prep.append(dict(root_id=row['root_id'],triangles=mesh.triangles,keyframes=len(mesh.keyframes),seconds=mesh.prepare_seconds));print('PREP',prep[-1],flush=True)
        with np.load(old['path'],allow_pickle=False)as z:x,y,prior=z['x'],z['y'],z['baseline']
        cms=[];mesh_cache={}
        for i,meta in enumerate(row['segments']):
            v=get_segment(a,i);path=v[3];e=lookup[meta['start_node'],meta['end_node']];sl=slice(e['start'],e['end']);keys=np.floor((x[sl,20:22].astype(np.float64)*10+path[0,[1,3]])/res).astype(np.int32);gt=y[sl];base=base_rotations(path[:,4]);origin=path[:,1:4]+base@translation;rot=base@rotation;cameras=np.concatenate([origin[1:],rot[1:].reshape(-1,9),path[1:,2:3]],1);lo=keys.min(0)-2;shape=keys.max(0)-lo+3;unknown=np.ones(tuple(shape),np.uint8);ij=v[0]-lo;valid=np.all((ij>=0)&(ij<shape),axis=1);at=ij[valid];unknown[at[:,0],at[:,1]]=0;p=prior[sl];free=(p[:,1]>p[:,2])&(p[:,1]+p[:,2]>=.2)
            if branch_current:
                from .range_history import ancestor_frames
                gi=int(a['graph_index'][i])
                if gi not in mesh_cache:
                    take=ancestor_frames(sensors['parents'],sensors['history_count'],meta['start_node'])
                    selected_tracks=[tracks[j]for j in take]if tracks is not None else None
                    mesh_cache[gi]=mesh_type(sensors['depth'][take],sensors['origins'][take],sensors['rotations'][take],sensors['timestamps'][take],path[0,0],res,device,stride,selected_tracks,plane_extension,v[0][v[1]==1])
                    mesh=mesh_cache[gi];prep.append(dict(root_id=row['root_id'],start_node=meta['start_node'],triangles=mesh.triangles,keyframes=len(mesh.keyframes),seconds=mesh.prepare_seconds))
                mesh=mesh_cache[gi]
            prior_free=free.copy();floor=x[sl,24].astype(np.float64)+path[0,2]
            if floor_support:
                supported,observed_floor=mesh.floor_support(keys,floor)
                floor=np.where(supported,observed_floor,floor);free|=supported
            if cspace_filter:
                clear=mesh.obstacle_distance(keys,path[0,2])>(.25+res/np.sqrt(2)+.008)
                free&=clear;prior_free&=clear
                if floor_support:supported&=clear
            if free_projection or floor_support:
                selected=np.flatnonzero(free);planar=np.array([[0,0,0],[-.49*res,0,-.49*res],[-.49*res,0,.49*res],[.49*res,0,-.49*res],[.49*res,0,.49*res]]);angle=np.arange(16)*2*np.pi/16;halo=np.column_stack([np.cos(angle)*(.25+res/np.sqrt(2)),np.zeros(16),np.sin(angle)*(.25+res/np.sqrt(2))]);offset=np.concatenate([np.concatenate([planar,halo])+[0,h,0]for h in [.025,.7,1.4]]);center=np.column_stack([(keys[selected,0]+.5)*res,floor[selected],(keys[selected,1]+.5)*res]);mesh.set_probes(keys[selected],center[:,None,:]+offset)
            code=mesh.blocked(unknown,lo,cameras,[keys],[np.arange(len(cameras),dtype=np.int32)])[0];geom=np.where(code<0,2,0);hybrid=np.where(code<0,2,np.where(code>0,1,0))if free_projection or floor_support else np.where(geom>0,2,np.where(free,1,0))
            predictions=[geom,hybrid]
            if floor_support:
                predictions.extend([np.where(code<0,2,np.where((code>0)&supported,1,0)),np.where(code<0,2,np.where((code>0)&prior_free,1,0))])
            pair=np.array([np.bincount(3*gt.astype(np.int64)+pred,minlength=9).reshape(3,3)for pred in predictions]);cms.append(pair);details.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],start_node=meta['start_node'],end_node=meta['end_node'],confusion=pair.tolist()))
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0)));print('ROOT',len(records),'windows',len(details),flush=True)
    scenes=defaultdict(list)
    for r in records:scenes[r['scene_id']].append(r['cm'])
    names=['range_geometry_blocked','range_hybrid']+(['range_floor_only','range_prior_only']if floor_support else[])
    cm=np.mean([np.mean(values,axis=0)for values in scenes.values()],axis=0);pooled=np.sum([r['confusion']for r in details],axis=0);result=dict(role='LEGAL SENSOR EXTENSION: current/past measured range images + current camera calibration + current graph + supplied base trajectory. No hidden mesh/navmesh or future camera. Full fixed pilot only when all31Roots/666windows present; not full-development acceptance.',roots=len(records),windows=len(details),stride=stride,tracked_filter=tracked_filter,free_projection=free_projection,plane_extension=plane_extension,branch_current=branch_current,floor_support=floor_support,gpu_mesh=gpu_mesh,cspace_filter=cspace_filter,hierarchical={k:extended_rates(c)for k,c in zip(names,cm)},pooled={k:extended_rates(c)for k,c in zip(names,pooled)},preparation=prep,records=details,seconds=time.perf_counter()-tick);save(OUT/('observed_range_stride%d_tracked%d_projected%d_plane%g_branch%d_floor%d_gpu%d_roots%d_cspace%d.json'%(stride,tracked_filter,free_projection,plane_extension,branch_current,floor_support,gpu_mesh,roots,cspace_filter)),result);print('SUMMARY',result['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--stride',type=int,default=1);p.add_argument('--tracked-filter',action='store_true');p.add_argument('--free-projection',action='store_true');p.add_argument('--plane-extension',type=float,default=0.);p.add_argument('--branch-current',action='store_true');p.add_argument('--floor-support',action='store_true');p.add_argument('--gpu-mesh',action='store_true');p.add_argument('--roots',type=int,default=31);p.add_argument('--cspace-filter',action='store_true');run(**vars(p.parse_args()))
