"""Fresh full-universe supervision and LEGAL planned camera-line origins."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import get_worker_info
from .fresh_data import FreshGroups,draw
from .diagnose import OUT,HF,read,source_rows,member
from .legal_camera import current_extrinsic,base_rotations
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def camera_lines(path,translation,resolution,lo):
    pose=base_rotations(path[:,4]);origins=path[:,1:4]+pose@translation
    selected=[1 if len(path)>1 else 0,len(path)//2,len(path)-1]
    return (origins[selected][:,[0,2]]/resolution-lo).astype(np.float32)


class RayFreshGroups(FreshGroups):
    def __init__(self,manifest):
        super().__init__(manifest)
        source,_=source_rows('train');source={r['root_id']:r for r in source}
        perroot={}
        for g in self.groups:
            rid=g['root_id']
            if rid not in perroot:
                legal=member(source[rid],'legal_root.json.gz');ob=legal['history'][-1]
                root_path=np.array([[ob['timestamp'],*ob['robot_pose']]],np.float64)
                translation,_=current_extrinsic(root_path,ob['camera_origin'],ob['camera_rotation'])
                packed=read(HF/'packed/pilot_fit'/(rid+'.json'));a=load_arrays(packed)
                lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
                perroot[rid]=(translation,a,lookup)
            translation,a,lookup=perroot[rid];g['rays']=[]
            for e,_ in g['strata']:
                path=get_segment(a,lookup[e['start_node'],e['end_node']])[3]
                g['rays'].append(camera_lines(path,translation,a['geometry'][0],g['low']))

    def __iter__(self):
        torch.set_num_threads(1);info=get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1
        epoch=self.epoch.value;order=np.random.default_rng(14317+epoch).permutation(len(self.groups))
        for gi in order[worker::workers]:
            g=self.groups[int(gi)];rng=np.random.default_rng(14317+epoch*1000003+int(gi));indices=[];weights=[];rays=[]
            for (e,populations),origins in zip(g['strata'],g['rays']):
                ids,w=draw(populations,e['end']-e['start'],g['root_windows'],rng)
                indices.append(ids);weights.append(w);rays.append(np.broadcast_to(origins,(len(ids),3,2)))
            ids=np.concatenate(indices);a=g['arrays']
            yield dict(maps=g['maps'],aux=g['aux'],mask=g['mask'],aux_weight=g['aux_weight'],x=a['x'][ids],extra=a['extra'][ids],y=a['y'][ids],indices=a['keys'][ids]-g['low'],weight=np.concatenate(weights)*g['scene_factor'],ray_origins=np.concatenate(rays))
