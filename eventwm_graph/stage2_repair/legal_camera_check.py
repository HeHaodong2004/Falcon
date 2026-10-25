"""Measure recorded future camera error of a strictly starting-camera proxy."""
from collections import defaultdict
import numpy as np
from .diagnose import OUT,HF,read,save,source_rows,member
from .legal_camera import proxy_cameras,current_extrinsic


def run():
    sets={};roots=[]
    for split in ['train','dev']:
        rows,_=source_rows(split)
        if split=='dev':ids=set(read(HF/'protocol.json')['pilot_selection_roots']);rows=[r for r in rows if r['root_id']in ids]
        else:rows=rows[:128]
        errors=[];matrices=[]
        for row in rows:
            legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');start=legal['history'][-1];future=[n['observation']for n in nodes];p=np.array([[o['timestamp'],*o['robot_pose']]for o in [start]+future]);o,r=proxy_cameras(p,start['camera_origin'],start['camera_rotation']);actual_o=np.array([q['camera_origin']for q in [start]+future]);actual_r=np.array([q['camera_rotation']for q in [start]+future]);oe=np.linalg.norm(o-actual_o,axis=1);re=np.max(abs(r-actual_r),axis=(1,2));errors.extend(zip(oe[1:],re[1:]));t,e=current_extrinsic(p,start['camera_origin'],start['camera_rotation']);matrices.append(np.r_[t,e.ravel()]);roots.append(dict(split=split,root_id=row['root_id'],frames=len(future),max_origin_error_m=float(oe.max()),max_rotation_element_error=float(re.max())))
        err=np.array(errors);sets[split]=dict(roots=len(rows),future_frames=len(err),origin_error_m_quantiles=np.quantile(err[:,0],[.5,.95,.99,1]).tolist(),rotation_element_error_quantiles=np.quantile(err[:,1],[.5,.95,.99,1]).tolist(),between_root_extrinsic_std=np.std(matrices,axis=0).tolist())
    save(OUT/'legal_camera_transform_check.json',dict(scope='Recorded FUTURE camera poses used only to score a proxy that accepts the LEGAL starting camera and supplied base poses. No hidden mesh or future observation is an input to proxy_cameras.',sets=sets,roots=roots));print(sets,flush=True)


if __name__=='__main__':run()
